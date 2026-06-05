# Final-push research + ranked build plan (2026-06-03)

> Durable record of the deep-research pass (`wf_89411d2c-fac`, 112 agents, 29
> sources, 135 claims → 25 adversarially verified 3-vote → 21 confirmed / 4
> killed) on the five catch-up levers + moat scout, **cross-checked against the
> live `paradigm/exec` tree**. North-star metrics: tokens/sec ↑, joules/token ↓.
> Anchor: M3 Pro 18 GB (~150 GB/s), Qwen2.5-3B-Instruct Q4_K_M, ~30.5 dec_tps /
> ~0.197 J/tok clean; llama.cpp ~49 tps same machine (1.6× gap).
>
> Where a source's numbers are from other hardware (CUDA 4090/5090, M2 Ultra
> 800 GB/s), it is tagged — those magnitudes DO NOT transfer to M3 Pro. The
> Type-1 kills in `reports/dead_levers.md` were fed to the research so it would
> not re-propose dead axes; none were.

## TL;DR — the reframe

**Most of the "final push" is wiring + measurement, not greenfield building.**
The competitive headline is unchanged (on raw batch-1 decode tps dismantle is
~0.62× llama.cpp and third behind MLX), but every major building block the
research points at **already exists in the tree** — so the effort estimates
collapse:

| Lever | Research said | Verified repo state | Real work |
|---|---|---|---|
| **① f16-scales default** | flip a default | `DISMANTLE_QWEN_PREDEC_F16SCALES` flag exists, default-off ([qwen_dense.rs:3507](../crates/dismantle-core/src/model/qwen_dense.rs)) | **one-flag flip** + re-baseline |
| **② fusion** (HEADLINE) | port llama PR #16220 | `add_rmsnorm_fused` exists ([common.metal:144](../crates/dismantle-core/shaders/common.metal)) but wired **only into deepseek**, not Qwen | wire into Qwen loop (opt-in first) |
| **③ GPU sampling** | build argmax kernel | `sample_argmax_f32` + `_tcb` + fused `gemv_f16_argmax_metal_pinned` all exist ([kernels/mod.rs:3013](../crates/dismantle-core/src/kernels/mod.rs)) | wire + verify bit-identical greedy |
| **⑤ flash / f16-KV** | write flash kernel | `mha_decode_flash_f32` exists ([mha.metal:152](../crates/dismantle-core/shaders/mha.metal)), default-off | default for long-ctx only |
| **⑥ CPU backend** | introduce seam | seam **landed**, Burn-shaped op-traits ([backend/metal.rs:153](../crates/dismantle-core/src/backend/metal.rs)); CPU ref path exists (`DISMANTLE_FORCE_CPU=1`) | add a CpuBackend **rung** |

The genuinely new build work is exactly two things: **(②) fuse the Qwen decode
loop** and **(⑥) a CPU backend rung**. Everything else is flip-default-and-measure.

## Per-lever verified survey

### Lever 1 — close the 1.6× gap (runtime / GPU-saturation) — **HEADLINE**
- **Batch-1 token-gen is memory-bandwidth-bound, so kernel *fusion* is the
  dominant tps lever** (cuts both memory traffic AND launch overhead). Primary:
  llama.cpp [discussion #17621](https://github.com/ggml-org/llama.cpp/discussions/17621)
  (collaborator am17an): *"TG is memory-bound … fusing kernels reduces memory
  traffic and kernel launch time."* **[CUDA-backend; the source itself warns the
  win shrinks at low bandwidth — M3 Pro is 6–7× lower BW than the 4090/5090 it
  was measured on, so the Metal magnitude is UNKNOWN until traced locally.]**
- **The copyable Metal mechanism:** llama.cpp Metal
  [PR #16220](https://github.com/ggml-org/llama.cpp/pull/16220) (ggerganov,
  verified by reading the merged diff) fuses `NORM/RMS_NORM + MUL + ADD` into a
  **single** dispatch via a compile-time `template<typename T, short F>`
  (F=1/2/3 = norm / norm+mul / norm+mul+add); the graph compiler pattern-matches
  the chain onto `kernel_norm_mul_add_f32` etc. "No new math" — mean/var/scale
  byte-identical, fusion is a templated write-back tail. (Nuance: the RMS_NORM
  variant pre-existed; #16220's novelty is unifying NORM with it.)
- **Op-chain map to fuse** (#17621): `MUL_MAT` following an `ADD`; and the gated
  activation `σ(W_gate·X)⊙W_up·X` reusing the X activation; `RMS_NORM` following
  a `MUL` and optionally an `ADD`.
- **MTLResidencySet** (llama PR #11427, macOS ≥15): rule out — dismantle's A/B
  was neutral and residency is Type-1-dead on unified memory.

### Lever 2 — portability / compute-backend seam
- **CubeCL** = single-source `#[cube]` JIT to 6 targets (CUDA/HIP/Metal/Vulkan/
  WebGPU/CPU), shape-adaptive heuristics ([burn.dev matmul blog](https://burn.dev/blog/sota-multiplatform-matmul/)).
  **Metal support is ALPHA** (documented M3 shared-memory bug `burn#4530`:
  40960 B requested vs Metal's 32768 B cap). **The claim that CubeCL's Metal
  backend is *materially weaker* due to less plane control was REFUTED (1-2)** —
  so do NOT assume a portability perf-discount; it's unquantified.
- **Burn's seam shape** ([Backend trait](https://burn.dev/docs/burn/tensor/backend/trait.Backend.html),
  v0.21): one supertrait bundling **7 op-traits** (Float/Bool/Int TensorOps,
  ModuleOps, ActivationOps, **QTensorOps**, TransactionOps). **Not dyn-compatible
  → monomorphized** (static generic, zero vtable cost, viral generics). Candle/
  mistral.rs use an enum-`Device` instead (dynamic, simpler, small cost).
- **Pattern to copy (resolves portability-vs-speed):** llama.cpp keeps
  **per-backend specialized kernels behind ONE seam + automatic CPU fallback**
  (`ggml_backend_sched`). Keep hand-tuned Metal as the Apple fast-path; generic
  CPU fallback makes a partial backend shippable day one.
- **Effort-per-rung: still unquantified** (both research passes failed; the one
  attempt to quantify the CubeCL-Metal gap was refuted). Needs a CPU-rung spike.

### Lever 3 — f16/bf16 activations + f16 KV + flash-decode
- **Confirmed: flash + f16/quantized-KV is an attention/long-context lever, NOT
  a short-context batch-1 tps win.** llama.cpp [PR #9735](https://github.com/ggml-org/llama.cpp/pull/9735)
  (Metal, Phi3-3B-Q4_K_M — same class as Qwen-3B): tg128 **30.97 → 31.00 t/s**
  (flat; 30.84 with q8_0 KV). Matches dismantle's prior internal "f16-KV is
  footprint-not-tps."
- **MLX precedent** [`mlx-qsdpa`](https://github.com/Thump604/mlx-qsdpa): fused
  inline-dequant quantized-KV flash, online-softmax, single dispatch, no score
  materialization — wins only at long context (1.71× @128K, **0.77× REGRESSION
  @4K**, 1.04× @1K). **[M2 Ultra 800 GB/s — does NOT transfer to M3 Pro; small
  unreplicated repo, directional only.]**
- **Value for dismantle = footprint / 32K-context headroom, not the gap.**

### Lever 4 — flip the f16-scales default off bit-identity
- **Shipping a quality-equivalent (not bit-identical) default kernel is the
  industry norm, not a quality risk.** Three independent authoritative sources:
  - NVIDIA TensorRT ([docs](https://docs.nvidia.com/deeplearning/tensorrt/10.12.0/inference-library/work-quantized-types.html)):
    *"results will not be bitwise identical … bit-level accuracy is rarely
    possible … (a·s)+(b·s) → (a+b)·s is a valid optimization."*
  - Google LiteRT/TFLite 8-bit [spec](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/quantization_spec):
    explicitly accepts non-bit-exact implementations within per-op tolerances.
  - FP non-associativity makes bit-identity generally unattainable (settled).
- dismantle's strict bit-identity gate on the default is **self-imposed**;
  llama.cpp/MLX gate nothing on bit-identity-to-reference. **Scoping caveat:**
  TensorRT's statement is about op-*reordering*; f16-scales additionally changes
  dequant scale *precision* (f16 vs pre-expanded f32) — slightly beyond reorder,
  so the local quality oracle must still confirm. Keep bit-identity as an
  **opt-in product feature** (see moat), not the default gate.

### Lever 5 — on-GPU sampling
- Literature came back **thin** — must be measured locally. Industry does on-GPU
  sampling to avoid the logit D2H copy; the per-token cost at vocab ~152K and the
  greedy-argmax tie-break determinism question are local microbench items.

### Moat scout
- **Energy IS measurable from user-space without sudo** — the moat opening.
  `zeus-apple-silicon` reads per-domain **GPU / GPU-SRAM / external-DRAM / ANE**
  energy in mJ via IOReport's "Energy Model" channel; `macmon` proves it's
  **Rust-bindable, no sudo**. **Method constraint:** values are MODEL-ESTIMATES
  (~1 mJ resolution); windows <10 ms are noise — aggregate over hundreds-to-
  thousands of tokens and divide, never per-token.
- **NO user-space GPU DVFS** (Asahi AGX docs: clocks are firmware/ASC-gated).
  So the only energy lever is **workload-shaping / race-to-idle** — which *is*
  Lever 1 (fusion → fewer GPU-active ms → lower J/tok). The two metrics couple.
- **Honesty correction:** "Apple decode-energy literature is empty" is going
  **stale** — mid-2026 papers now measure it (arXiv 2605.00519 "Silicon
  Showdown" on M3 Pro/Ultra; arXiv 2512.03024). The defensible moat narrows to:
  *no engine ships **in-process** per-domain J/tok instrumentation + opt-in
  bit-identical deterministic inference as product features.* Still real.
- **Rust-native:** Cloudflare's Infire ([blog](https://blog.cloudflare.com/cloudflares-most-efficient-ai-inference-engine/))
  validates the *thesis* (Rust to escape Python) — **[but H100, batched serving,
  vs Python overhead, NOT Apple batch-1 and NOT vs C++ llama.cpp].** The honest
  Rust edge: zero-Python single-binary distribution, embeddability, memory
  safety, WASM/edge + the energy instrument + opt-in determinism + the existing
  stateful prefix-cache (~84% prefill elision). Not a speed edge.
- **ANE** stays dead for decode. Nothing changed.

## Ranked build plan (certainty-per-effort, cheapest-oracle-first)

| # | Lever | Payoff | Effort | Risk | Gate / oracle |
|---|---|---|---|---|---|
| **①** | f16-scales default | **+9.3% tps, −1.4% J/tok** (measured) | XS (flag flip + re-baseline) | low — industry norm | `quality_oracle.sh` (Claude-open OK) → flip |
| **②** | fuse Qwen NORM+MUL+ADD | high, **unquantified @150 GB/s** | M | med (parity per kernel) | **Metal System Trace** (clean room) BEFORE default-on; build opt-in now |
| **③** | GPU greedy argmax | unknown — measure D2H+argmax cost | S (kernel exists) | low-med (tie-break determinism) | paired microbench; bit-identical greedy |
| **④** | in-process per-domain J/tok | publishable moat; couples to ② | S-M | low (estimate caveat) | wrap clean-room run, sanity vs 0.197 J/tok |
| **⑤** | flash + f16-KV default | **not tps** — footprint/long-ctx | M (kernel exists) | low | long-context A/B (not short) |
| **⑥** | CPU backend rung | structural; **zero tps** | L (rung, seam landed) | med (perf gap unquantified) | spike ONE rung, measure LOC first |

## Metal System Trace protocol (Lever 2 gate)

1. **Quit Claude** (contamination 4–5×). Fixed prompt, temp=0, fixed seed,
   ~256-token decode, identical Qwen2.5-3B-Q4_K_M for both engines.
2. Instruments → **Metal System Trace**. Attach separately to (a) `dismantle`
   release binary, (b) `llama-cli` Metal build. Capture GPU track (command-buffer
   boundaries, per-dispatch compute intervals) + CPU encode track.
3. **Per steady-state decode token, extract:** (i) dispatch **count** (dismantle
   ~180 vs llama — the delta is the fusion opportunity); (ii) GPU-busy fraction
   (~76% → ~24% idle vs theirs); (iii) inter-dispatch **gap distribution** (one
   big stall or many small gaps?); (iv) commit→GPU-start latency.
4. **Attribution:** many small gaps + high count → **fusion (②) is the lever,
   port #16220**. One large per-token commit/wait bubble → the lever is
   **command-buffer structure** (persistent/compiled CB, `MTLIndirectCommandBuffer`
   for the repeated decode step), not fusion. Longer GPU intervals per-FLOP than
   llama → kernel-level (but Q4_K GEMV is Type-1-optimal, so unlikely).

## What the literature CANNOT answer → measure on our M3 Pro
- (a) The actual Metal fusion payoff magnitude at 150 GB/s (all quantified wins
  are CUDA/high-BW). → System Trace + fused-kernel paired A/B.
- (b) The ~24%-idle attribution (dispatch-count vs commit/wait vs scheduling). →
  the Instruments trace is the only settling measurement.
- (c) Per-token logit D2H + CPU-argmax cost at vocab 152K / pruned 32K. → paired
  microbench of the existing GPU-argmax kernel.
- (d) Per-domain GPU-vs-DRAM J/tok for dismantle's decode loop. → IOReport
  aggregate over a long clean-room run. **This is the energy moat.**
- (e) f16-scales quality-equivalence at long context on Qwen-3B specifically. →
  local quality oracle.

## Refuted at verification — do NOT rely on
- CubeCL-Metal-is-materially-weaker (1-2) → don't assume a portability discount.
- CUB GPU-sort slower-but-memory-win (1-2).
- TokenPowerBench has zero Apple coverage (0-3) → Apple energy papers DO exist.
- macpow gives direct energy deltas (1-2) → it's model-estimate; prefer zeus.

---

## Implementation status (2026-06-03 pass — Wave-1 swarm `wf_5d69b392` + serial apply)

The swarm's read-only audit corrected **three of five** briefs against the live tree.

- **① f16-scales default — TRIED then REVERTED `e613dde` (FAILED the quality gate).**
  The flip (`b417495`) built clean, passed kernel parity (`q4k_predec_f16s_parity`
  rel-L2 < 1e-2) + 94/9 lib tests, and paired ABBA confirmed the tps win
  (**B/A=0.917, ~+9%**). BUT the corpus quality oracle (`quality_oracle.sh`, 24
  diverse prompts × 48 tok, f16-default vs f32-opt-out) measured **token-identical
  0.792 (gate ≥ 0.90) and corpus drift 11.46% (gate ≤ 5%)** — 5/24 prompts diverge,
  up to 35% prose / 18% math (high-entropy; code/lists/sql stay identical). This is
  the known f16-rounding signature (cf. q4k_fast_divergence, w4a8_corpus_quality) —
  exactly why f16-scales shipped as an opt-in "mild quality trade," not a default.
  **Correctness gates before performance**, so the default stays bit-identical
  f32-scales; f16-scales remains opt-in via `--profile fast` for code-shaped
  workloads (where divergence ≈ 0). **Lesson:** the research's "industry ships
  quality-equivalent" principle is real, but the dismantle-specific oracle is the
  binding gate — and it says f16-scales is NOT within this repo's equivalence bar.
  Oracle JSON: `reports/quality/oracle_f32scales_optout.json`.
  *Re-attempt only with a quality fix (e.g. per-block f16 scale + f32 dmin, or a
  selective-precision scheme that keeps the high-entropy logits f32).*
- **④ per-domain energy instrument — LANDED `a90fe80`.** `phase_joules.sh --domains`
  emits GPU + DRAM J/tok from macmon `ram_power` (no dep, sudo-free), gated so
  default output is byte-identical. Smoke-test (Claude open): GPU 0.080 / DRAM
  0.042 J/tok populate; ratio ~1.9:1 is the contamination-robust signal.
  GPU-SRAM not exposed on M3 Pro. *Absolute split needs a clean room.*
- **③ GPU sampling — ALREADY SHIPPED (no-op).** The default greedy TCB path already
  runs GPU argmax (`sample_argmax_f32_tcb`, qwen_dense.rs:4933/5011) and reads back
  4 bytes; CPU argmax is temp>0 only. Tie-break already matches (lowest index,
  sample.metal:47/69). The brief conflated the CPU-hybrid `forward_token` path
  with the default. No work.
- **② fusion — KILLED/LOGGED (Type-1 on this path).** The Qwen hot path *already*
  fuses both add+RMSNorm sites and gate+up (and tail-hoists ~73 dispatches/tok);
  the llama.cpp PR #16220 port has no standalone MUL/ADD node to absorb. Logged in
  `reports/dead_levers.md` (Phase 2.2 entry, 2026-06-03 update) with the silu+down
  Type-2-tiny reframe + its `ab_lever.sh` oracle. **Not built** (exhausted regime).
  **Strategic upshot:** fusion is spent → the 1.6× gap is NOT dispatch-count; the
  Metal System Trace is now the *sole* path to it.
- **⑥ CPU-backend rung — SPEC + LOC measured (build deferred).** The oracle asked
  for the LOC first; it is: **~340 LOC** for a compile-stub `ComputeBackend` rung
  (`backend/cpu.rs`: `CpuBuffer`/`CpuRecorder` + 10 op-traits, real add/silu/rmsnorm/
  rope/F16+F32-gemv/mha/kv/embed/argmax, `Err`-stub q8-norms/quant/Mla/Moe), or
  **~135 LOC** for the cheapest single-op (elementwise-add) spike echoing the
  Metal seam-add proof. The CPU *compute* is already parity-green (`forward_token`
  + `cpu_backend_parity.rs` 12/12). Critical seam facts: `trait Backend: Sized` +
  GAT `Recorder<'a>` ⇒ **not object-safe** (no `Box<dyn Backend>`); `Router` is
  **dormant** (not wired into decode). So landing the *type* is clean, but routing
  real decode through it is a separate, larger change — deferred (zero-tps,
  structural). Build it when the portability rung is the active goal.

**Still gated on you / the clean room:** (a) the **Metal System Trace** diff
(needs Claude quit + Instruments + a `llama-cli` Metal build) — now the decisive
next step for the 1.6× gap; (b) clean-room **absolute** re-confirmation of ④'s
per-domain split (`phase_joules.sh --domains --tokens 512`, Claude quit).

---

## Throughput investigation (2026-06-03 follow-up) — "harness the CPU?" + maximize throughput

Two investigation waves (`wf_b6d6913b` survey + `wf_10625e6d` build) + an empirical close.

- **CPU-harness for tps — NO-GO Type-1 (empirically closed).** On the one shared
  ~150 GB/s unified bus, single-stream decode time = bytes/token ÷ bandwidth —
  neither term changes by *who* reads; the GEMV runs ~56% of peak with **0.0 ms
  inter-dispatch idle**, so a concurrent CPU read contends, not adds. Aggregate:
  CPU decode measured **0.06 dec_tps** (`DISMANTLE_FORCE_CPU=1`, Qwen-3B) — it
  re-dequantizes Q4_K→f32 every token (~28.9 GB/token, **15× worse bytes/token**
  than the GPU). llama.cpp `-ngl` split is a memory-fit workaround, not a
  throughput win (web-confirmed). Logged: `dead_levers.md` (CPU+GPU pipelining,
  2026-06-03 update).
- **Continuous batching — GO, the real aggregate prize; effort L→M.** `batch_ceiling.py`
  predicts realistic **~3.5–5.6× aggregate at B=8** (KV doesn't re-saturate until
  B≈26–102 — GQA n_kv=2 makes KV ~1–4% of the read). The shipped **v3w B=8 GEMM**
  (`gemm_q4_k_m_batched_v3w`, quant.metal:1683) already reads each weight once and
  applies to B columns — sequence-agnostic, exactly what multi-stream decode needs.
  The serve scheduler/driver/sampler control plane is built but unwired. **Remaining
  build (M, its own attended task):** (1) per-slot KV cache (slot-strided, ~0.6 GB
  @ B=8, no paging needed for prototype); (2) ONE new kernel `mha_decode_f32_batched_multiseq`
  (per-slot position array + per-slot KV base — a modest edit of `mha_decode_f32_batched`);
  (3) per-slot KV append; (4) a real `Engine::forward_tokens_batched` for Qwen + serve
  wiring. Parity: each batched column == the same prompt decoded one-at-a-time (b3sum).
  Build against f32 KV first.
- **f16-scales recovery — NO-GO Type-1 (offline oracle).** `oracle_f16scales_precision.py`
  over all 216 Q4_K tensors: f16 rounding error is **uniform** (ds ~55% / dm ~45% of
  variance, independent; 1.29× layer spread). Asymmetric f32-dmin removes only ~26%
  of drift (→ ~8.5%, still > 5% gate); no hot subset for selective precision. f16-scales
  stays opt-in. (The one unexplored path is a *different* representation — bf16 scales,
  or f16 + f32 correction — a new lever, not a reframe. Separately, the gate-realism
  question — is greedy-token-identity over-strict? — remains open behind a `--dump-logits`
  build.)
- **Metal Trace harness — LANDED `91e3446`.** `tools/bench/mst_diff.sh` + `mst_gap.py`
  (CPU-validated) + `ProdCbGpu` raw-timestamp capture. Ready to run Claude-quit; decides
  whether llama's `mul_mv` sustains higher GiB/s/call (the sole single-stream reframe;
  adverse prior). The `gpu_start_ns/gpu_end_ns` capture also gives the production
  inter-dispatch gap without Instruments.

**The honest throughput map:** single-stream batch-1 is structurally tapped (every
axis Type-1 dead except the adverse llama-GEMV-technique reframe, which the Trace
harness now gates). The one large live lever is **GPU continuous batching for
aggregate/serving tps** (~3.5–5.6× at B=8), and its hard part (the weight-amortizing
GEMM) already ships as v3w — the remaining work is the multi-seq KV/attention layer,
an M-effort attended build.

---

## Review wave (2026-06-03) — pre-push findings + forward specs (`wf_d81c7e7a`)

A read-only review/spec swarm over the 14-commit continuous-batching build. **The
shipped DECODE code is VERIFIED-OK** for what it's tested on (the multi-seq MHA
softmax + per-slot indexing, the divergent-position KV-offset arithmetic in the full
forward, per-slot append — all confirmed correct by trace). **The continuous-batch
path has ZERO non-test callers — the HTTP server still does one-request-per-mutex
`engine.generate`** — so the findings below are **latent** (not corrupting anything
today) and detonate only when the path is wired to serving (the deferred HTTP-loop
work). The review caught them *before* that wire-up.

### Serving-path blockers (fix as part of the HTTP-loop build, NOT shipped bugs)
1. **No prefill — the multi-seq path is decode-only** (one KV append/slot/call). A
   served request would decode against an empty KV prefix. Needs a multi-seq prefill
   (or decode-from-0 per request). CONFIRMED-BUG for serving.
2. **Arena indexed by compacted batch-position, not stable slot-id.** Two detonations:
   (a) slot evicts → ready-set compaction shifts indices → a slot reads *another
   slot's* KV (cross-contamination); (b) B grows → `forward_tokens_multiseq_logits`
   reallocs `multiseq_arena` → fresh zeroed buffers → **all in-flight KV wiped**.
   Fix: thread a stable `slot_id`/region through `forward_multiseq_batched`, allocate
   the arena **once at `max_batch` (never realloc on growth)**, zero-on-release.
   "F0.3 interim" (fixed-`max_batch` arena) is **S** and unblocks everything.
3. **`MULTISEQ_CTX=2048` cap** is silent-until-error + uncoordinated with the
   scheduler (RSS at B×2048 per-slot full KV). Paged KV removes it.
4. **Test gap:** the equivalence test only covers *lockstep positions + constant B*.
   Needs a **divergent-position + varying-B + slot-churn** parity test (admit 4,
   evict one at EOS, admit a 5th; each survivor's tokens == its solo decode). This is
   the test that would catch #1 and #2.

### Aggregate-tps optimization (2.42–2.57× → ~3.5–5.6×), ranked
- **RANK 1 (M) — GPU-batched LM head** (the dominant un-amortized cost: B sequential
  CPU full-vocab matmuls, ~622 MB f16 read ×B). **No batched f16 GEMM exists in-tree**
  → either Q4_K-requant the LM head (reuse `Q4K_LMHEAD`, opt-in to keep the anchor
  parity test green) or build one. Template: `forward_tokens_verify` FAST path
  (`gemm_q4_k_m_batched_v3w` over the pruned Q4_K head). Biggest single jump.
- **RANK 2 (S–M) — batch the per-slot embed + layer-0 rmsnorm + RoPE** (4B → 4
  dispatches/step). Bit-identical refactor.
- **RANK 3 (M) — single batched KV-append kernel** (2B → 1; removes ~574
  dispatches/step at B=8). Bit-identical.
- **RANK 4 (L) — multi-seq prefill + arena lifecycle** = serving prerequisite (#1/#2),
  not a perf lever. Already-batched (verified, no work): biases, v3w projections, silu_mul.

### Paged-KV + concurrent-HTTP serving spec
- **Paged KV** (lifts the 2048 cap + cuts RAM): shared page pool + per-slot block
  tables; the bulk of the effort is the MHA-kernel rewrite (block-table indirection
  replacing the fixed `kv_slot_stride`). Bounded if ctx ≤ ~6K (materialized scores).
- **HTTP loop:** built = slot manager + `decode_ready_once` + per-slot `Sampler`
  (verified-OK, seeded per slot); missing = the admission loop, per-slot SSE
  streaming, and the mutex→loop change. Depends on F0.3 (fixed-arena) first.

### Bench-harness audit (so the next clean window isn't wasted)
- **P0 (do FIRST in the clean window):** `mst_gap.py` STAGE 3/4 is **untested against
  a REAL Instruments export** (only synthetic XML; zero fixtures). Capture one tiny
  `TOKENS=8` MST, run `mst_export.sh` + `mst_gap.py` on it, confirm the parse —
  *before* spending the expensive 2-engine 256-tok capture. Schema (not numbers) is
  what's at risk, so this can be done attended/short.
- **P1 — FIXED (`da4acb4`):** fail-loud on no-measurement + queue failure-signature scan.
- **P3 — `mst_diff.sh` STAGE 2 llama-flag fragility** (`-no-cnv`/`--seed` drift): make
  the llama capture non-fatal (probe `--help`) so a flag drift doesn't waste the
  already-paid dismantle trace.
- **P4 — macmon `--domains` `ram_power` absence** → prints `0.0000` silently; preflight
  the field + print "DRAM unavailable" honestly.

### Fixed 2026-06-04 ("go fix all" — all BUGS fixed; feature-builds remain)
**All bug-fixes from the review are landed + validated:**
- **#2 arena/slot-id — FIXED `f02101e`** (the catastrophic one). KV is now keyed by a
  STABLE per-slot region (the slot id), not the compacted dispatch index: the MHA
  kernel takes a `regions[]` buffer, the append writes to `regions[bi]`, and the arena
  is fixed-capacity (8 regions, allocated once, NEVER reallocated on B-growth — kills
  the realloc-wipe). The driver passes `regions = slot ids`. Both detonations (evict
  cross-contamination + grow-wipe) are closed.
- **Test gap — FIXED** `tests/multiseq_churn_parity.rs`: reproduces a slot-1 eviction +
  index compaction; survivors stay byte-identical to their solo decode. Passes.
- **Bench P1 — FIXED `da4acb4`** (fail-loud + queue signature-scan); **P3/P4 — FIXED
  `c65b870`** (llama capture non-fatal → dismantle-only; honest "DRAM unavailable").

**Still FEATURE-BUILDS (not bug-fixes — own focused sessions, each parity-gated):**
- **#1 multi-seq prefill** — the path is still decode-only; a served prompt needs its
  KV prefilled. Prereq for the HTTP loop. (L)
- **#3 paged-KV** — lift the `MULTISEQ_CTX=2048` cap + cut per-slot RAM (MHA-kernel
  block-table rewrite). (L)
- **HTTP concurrent-batch loop** — admission + per-slot SSE + mutex→loop. (M–L)
- **Aggregate-opt R1–R3** — GPU-batched LM head (R1, M, the big lift) + batch the
  per-slot dispatches (R2/R3) to push 2.42× → ~3.5×. (optimization)
- **P0 mst_gap real-export parse** — validate on a tiny captured trace first (clean
  window). (pre-check)

---

## R1 (GPU-batched LM head) — APPLIED + VALIDATED + COMMITTED `8aba79e` (2026-06-04)

**What:** the aggregate-opt RANK-1 lever — replace the B sequential CPU full-vocab f16
matmuls in `forward_tokens_multiseq_logits` (qwen_dense.rs) with ONE GPU-batched Q4_K
GEMM over all B slots (weight read once, broadcast across B columns). Opt-in behind
`DISMANTLE_QWEN_Q4K_LMHEAD=1` — the SAME flag/buffer the verify FAST path uses.

**Approach (reuse, NO new kernel):** the flag-ON branch calls the identical
`gemm_q4_k_m_batched_v3w_pinned_tcb` that `forward_tokens_verify` (qwen_dense.rs:6187)
drives, over the SAME `arena.x_norm_buf_batch` the multiseq stack just wrote ([B,h]
row-major f32 = the v3w x layout, no transpose), into a fresh `ctx.new_buffer(B*vocab*4)`,
then slices B full-vocab rows. **Prune-trap correction (the wave caught it):** the
multiseq contract returns FULL-vocab logits and `forward_tokens_multiseq` argmaxes the
index DIRECTLY as a token id (no remap), so the GPU path uses the FULL-vocab
`self.lm_head_q4k_buf` (rows=vocab) — NOT the pruned head verify uses. Using the pruned
head would shorten the return vectors and emit pruned indices as token ids. Gate mirrors
verify: `env_on && lm_head_q4k_buf.is_some() && h%256==0 && B∈1..=8`; any miss falls
through to the unchanged CPU else-branch.

**Default stays byte-identical:** the edit only PREPENDS the guarded branch + early
return; the CPU per-slot loop is untouched as the flag-OFF else. The anchors
(multiseq_decode_parity, multiseq_churn_parity) `remove_var` the flag → always hit the
CPU path → stay green.

**Built via wave `wf_99664804`** (9 agents: 5 grounded reads → 1 synthesis → 3
adversarial-review lenses; all 3 returned ship / zero bugs / applies-cleanly /
default-bit-identical). **Re-verified independently** against the live tree before
applying (never trust the agent's "passed"): old_code anchor unique; `lm_head_q4k_buf:
Option<PinnedBuffer>` (qwen_dense.rs:207); single-token path (5022-5031) proves the
full-vocab reuse + byte formula `vocab*(h/256)*144`; v3w call+commit pattern (verify
6187-6191); `new_buffer→Buffer ≡ PinnedBuffer` (metal/mod.rs:381 alias) so the scratch
buffer type-checks; `x_norm_buf_batch` is a PinnedBuffer; borrow check = all immutable
self-borrows (mirrors verify). Both edited files PARSE-clean (rustfmt edition 2021) — a
syntax backstop, NOT a type/borrow check.

**STATE: VALIDATED + COMMITTED `8aba79e` (engine, 2 files +237).** The CPU wave cleared and
the deferred build + parity all ran green (Claude open — parity is contamination-robust):
- `cargo build --release --workspace` — Finished 45.83s, EXIT 0, no new warnings from the edit.
- `cargo test --release --workspace --lib` — 5 + 94 + 9 pass, 0 failed.
- `multiseq_decode_parity` + `multiseq_churn_parity` (flag-OFF anchors) — pass (6.99s / 8.31s);
  the default path is byte-identical (the edit only prepends a guarded branch + early return).
- `multiseq_q4k_lmhead_parity -- --ignored --test-threads=1` (flag-ON R1 gate) — 2/2 pass (9.81s):
  B=4 batched argmax == solo over 4 lockstep steps AND at divergent positions incl. pos 2047;
  full-vocab length asserted. The tie-flip open-risk did NOT materialize (v3w computes each
  column's reduction independently → batched==solo is bit-identical).
Files: `crates/dismantle-core/src/model/qwen_dense.rs` (the branch in `forward_tokens_multiseq_logits`)
+ NEW `crates/dismantle-core/tests/multiseq_q4k_lmhead_parity.rs`.

**Open risks carried:** (1) flag-ON logits are quant-noise-different from flag-OFF f16
(~1e-3) — the gate is batched-vs-solo argmax (BOTH flag-ON), NOT ON-vs-OFF bit-equality;
the OFF default stays the golden f16 path. (2) the added 4th seed (151643) / divergent
slots could in principle tie-flip an argmax under near-ties; but v3w computes each
column's reduction independently, so batched==solo is bit-identical per the existing
multiseq_decode_parity precedent — no flakiness expected; if it shows, drop to the
known-stable tri-seed.

**Next (each its own parity-gated pass):** R2 (batch per-slot embed/layer0-norm/RoPE,
4B→4 dispatches) + R3 (single batched KV-append, 2B→1), then re-measure aggregate (was
2.42×; R1+R2+R3 target ~3.5×) in a clean-ish window.

---

## R2 + R3 (batch the per-layer per-slot dispatches) — APPLIED + VALIDATED + COMMITTED (2026-06-04)

The two remaining aggregate-opt build levers ("top 2"), each a bit-identical refactor,
parity-gated + committed:

- **R2 — batched RoPE `88a00ef`.** New kernel `rope_f32_batched_multiseq` (common.metal)
  + wrapper replaces the per-slot `rope_q_f32_inplace` loop (2B dispatches/layer → 2),
  reading a per-slot `positions[]` buffer. RoPE is elementwise → BIT-IDENTICAL. Parity
  `rope_batched_multiseq_parity`: max_abs_diff == 0 vs the per-slot loop for the Q width,
  the K width (GQA), and B=1.
- **R3 — batched KV scatter-append `113a9a8`.** New kernel `kv_scatter_append_multiseq`
  (common.metal) + wrapper replaces the per-slot `memcpy_f32_off` loop (2B/layer → 1,
  K+V together), reading per-slot `regions[]`/`positions[]`. Pure copy → BYTE-IDENTICAL.
  Preserves the stable-slot-region keying. Parity `kv_scatter_append_multiseq_parity`:
  K+V caches identical vs the per-slot loop with churned (non-identity) regions, divergent
  positions, and a non-zero layer offset.

At B=8 over 36 layers, R2+R3 together drop ~2000 sequential per-slot dispatches/step
(2×576 RoPE + 576 append → ~144). Each validated: release build clean; the new-kernel
parity tests pass bit-identically; the integrated `multiseq_decode_parity` +
`multiseq_churn_parity` stay byte-identical; the R1 Q4_K LM-head gate (2/2) + 94/9/5 lib
tests stay green.

**Stack state:** every PER-LAYER per-slot loop on the Qwen-Q4_K hot path is now batched
(projections / biases / silu_mul / FFN / MHA / add_rmsnorm were already batched; RoPE +
KV-append now too). The ONLY per-slot loops left are the two PER-STEP ones — embed
(B/step) and layer-0 pre-rmsnorm (B/step) — together ~14 dispatches/step at B=8, ~0.7% of
what R2+R3 removed. Deferred as a negligible micro-opt that should itself be gated on the
bench showing per-step dispatch overhead matters.

**Only benching left (the axis is build-complete):** re-measure the B=8 aggregate
(`batch_aggregate_bench.sh` / `multiseq_aggregate_bench`, paired flag-ON vs flag-OFF —
contamination-robust, Claude-open OK) to confirm R1+R2+R3 moved 2.42× toward the ~3.5×
ceiling; then the clean-room absolutes. Commits `8aba79e` (R1) + `88a00ef` (R2) +
`113a9a8` (R3), local, 23 ahead.

### Aggregate preview (2026-06-04, Claude-OPEN — ratios/delta valid, absolute inflated) `6137883`

The aggregate bench now runs flag-OFF + R1-ON and prints the R1 delta. Contamination-
robust preview (the ~4–5× inflation cancels in ratios/deltas):

| B | flag-OFF agg scaling | R1-ON agg scaling | R1 delta (ON/OFF) |
|---|---|---|---|
| 1 | 1.00× | 1.00× | ×1.81 |
| 4 | 1.28× | 3.15× | ×4.47 |
| 8 | **1.57×** | **4.91×** | **×5.66** |

**R1 (GPU-batched Q4_K LM head) is the DOMINANT aggregate lever.** flag-OFF, the 8
sequential CPU full-vocab f16 matmuls dominate the B=8 step (per-step ~1239 ms), capping
aggregate scaling at 1.57×. flag-ON, one weight-amortizing Q4_K GEMM replaces them
(per-step ~219 ms — ×5.66 faster), and B=8 aggregate scaling jumps to **4.91×**, inside
batch_ceiling.py's 3.5–5.6× window. R2+R3 (baked into both configs) cut the GPU-stack
dispatch overhead; their effect shows in the absolute per-step time, and they actually
*lower* the flag-OFF ratio (1.57× vs the historical 2.57×) because a faster GPU stack
exposes the CPU LM head as the dominant bottleneck — a ratio artifact, NOT a regression.

**Implication:** the aggregate/serving win REQUIRES R1 flag-ON
(`DISMANTLE_QWEN_Q4K_LMHEAD=1`); the default stays flag-OFF (bit-identical anchor). The
clean room pins the ABSOLUTE aggregate tokens/sec — but the *increase* is already measured
here (contamination-robust). It is a real, large increase to be measured, not a diagnostic.

## CLEAN-ROOM RESULTS (2026-06-04, Claude.app quit; M3 Pro / macOS 26.5; HEAD `f7ab8c0`)

First clean run of `FAST=1 clean_bench_queue.sh`. `dec_tps 32.65` confirms the room was clean.

**Anchor (§B): single-stream decode = 32.65 dec_tps** (128 tok, greedy) — tracks the ~31 recent
anchor (+5.3%), not ~39 (−16.3%). R1/R2/R3 don't touch single-stream (expected). Q3 byte-cut (§A)
**NO-GO** reconfirmed: f32-predec-Q3 = 34.5 GB/s = 23% of peak → QTIP is the only byte-cut path.

**Energy (§2): clean per-domain baseline (256 tok) = 0.1956 J/tok — GPU 0.1773 / DRAM 0.0848**
(≈2.1:1). The moat number. CAVEAT: the f16-KV pass ran 512 tok (0.2423 J/tok) vs the 256-tok
baseline → the compare is CONFOUNDED by thermal ramp (longer run → higher avg power → higher
J/tok), so "f16-KV worse" is NOT a valid read; re-run both at the SAME N. (Also §C at 128 tok =
0.1566 under-reports vs 256-tok 0.1956 — J/tok rises with run length until thermal equilibrium;
≥256 is the faithful floor, 512+ for a publishable steady-state.)

**Aggregate (§4) — the payoff, CLEAN:**

| B | flag-OFF tps / scaling | R1-ON tps / scaling | R1 delta (clean) |
|---|---|---|---|
| 1 | 7.79 / 1.00× | 9.54 / 1.00× | ×1.23 |
| 4 | 15.91 / 2.04× | 30.98 / 3.25× | ×1.95 |
| 8 | 19.12 / **2.45×** | **47.96 / 5.02×** | **×2.51** |

- **R1+R2+R3 (flag-ON) B=8 = 47.96 tok/s, 5.02× scaling** — inside batch_ceiling's 3.5–5.6×.
- **R1 clean delta = ×2.51 at B=8, NOT the ×5.66 the Claude-open preview showed.** The preview
  overstated R1: Claude's CPU contention slowed the CPU-LM-head (flag-OFF) arm ~3× while barely
  touching the GPU (flag-ON) arm, so the paired delta did NOT cancel. **LESSON: paired A/B deltas
  are contamination-robust ONLY when both arms share a bottleneck class (both GPU or both CPU);
  R1 flag-ON/OFF crosses CPU↔GPU, so it MUST be measured clean.** (Within-config scaling: flag-ON
  4.91→5.02× was robust ✓; flag-OFF 1.57→2.45× was not — same cause.)
- **Honest serving win:** multiseq B=8 flag-ON (47.96) vs running the fast single-stream path
  sequentially (32.65) = **1.47× more aggregate throughput** (trading per-stream latency: 6.0 vs
  32.65 tps/stream). The 5.02× is scaling vs the *unoptimized* multiseq-B=1 (9.54), itself 3.4×
  below the single-stream path (32.65) — closing that B=1 gap (fused single-path kernels + the
  deferred per-step batches inside multiseq) is the NEXT aggregate lever; it lifts the whole curve.

**Trace (§3) FAILED — tooling, not dismantle.** `xctrace record` exits non-zero through the
`env+nice+taskpolicy` wrapper on macOS 26.5 even though the bench it launches succeeds (reproduced
standalone: exit 0, valid stats, dispatch_count 7392/8tok). Fix: make the dismantle capture
non-fatal + verify the `.trace` file exists (mirror the llama P3 non-fatal pattern), and/or drop
the wrapper under `xctrace --launch`. Needs a clean window to validate.

## Aggregate-max wave + C1 (2026-06-04) — the profile is the verdict; C1 landed; v3w is the 80%

Wave `wf_60586084` (12 agents) profiled the multiseq-B=1 gap + designed 4 levers. The PROFILE is
the headline (evidence: B-scaling fit `per_step_ms = 96 + 8.8*B`, predec ON/OFF null result, CPU
microbenches — all cross-consistent to 105ms):
- **~80% of the 105 ms multiseq-B=1 is the layer projection GEMMs running `gemm_q4_k_m_batched_v3w`
  at low B** — under-occupied (8 rows/TG, no 2-row-ILP), **DRAM-latency-bound NOT bandwidth-bound**
  (predec ON/OFF identical at B=1, vs single-stream's +45% without predec). gate+up also unfused
  (+36 GEMMs/tok). This pass is **B-independent** (the weight read) → it caps B=8 too → **the
  aggregate lever.**
- LM-head 2nd commit ~9%; per-seq marginal ~8% (the good amortizing part → B=8 scales 5x); per-step
  embed/layer0 ~3-4%; CPU argmax+to_vec **<0.5%**. So it's ~99% GPU-kernel-efficiency, ~1% CPU —
  argmax/copy is NOT the problem.

**LANDED: C1 single-TCB tail `8eaf627`** — the stack returns the live TCB; the R1 LM-head GEMM
appends into it → ONE commit/step (was 2). Bit-identical (anchors byte-identical, new test green,
94/9/5 lib). ~5-10 ms/step removed; per-step so it compounds into the aggregate.

**Marginal (profile-confirmed, NOT applied — low ROI):** A3 batch embed/layer0 (~0 ms B=1, 1-3 ms
B=8; ship-verdict, bit-identical, available on request); batched GPU argmax (<0.5% B=1; fix-then-ship);
A7 hoist allocs (~0.5 ms; design incomplete).

**THE BIG LEVER (the 80%, which the wave's design agent FAILED to produce): v3w projection-GEMM
efficiency at low B.** Two routes, each a focused parity-gated pass:
- **(a) MMA swap** — reuse the existing parity-tested `gemm_q4_k_m_batched_v3w_mma` in the multiseq
  stack. Easier, but MMA underfills at low B (may regress B=1) AND is numerically ≠ the single-stream
  gemv (reduction-reorder, atol 1e-3) → the `multiseq_decode_parity` ANCHOR (B=1 multiseq == single-
  stream) could flip an argmax → needs the anchor relaxed + a paired clean measure to confirm the B=8
  gain. Uncertain.
- **(b) v3w kernel rewrite** — port the tuned single-token gemv's 16-rows/TG + 2-row-ILP DRAM-latency-
  hiding into `gemm_q4_k_m_batched_v3w[_predec]`. The real fix: lifts the weight-read efficiency at
  ALL B → shrinks the 96 ms fixed pass → lifts the B=8 aggregate directly (47.96 → potentially ~80+).
  A real Metal-shader pass, gated by `q4k_batched_gemm_parity` + the multiseq anchors.

This is the genuine path past the 47.96/5.02× baseline. It deserves its own focused session, not a
rushed end-of-turn change (route (a) risks the anchor; route (b) is multi-step shader tuning).
