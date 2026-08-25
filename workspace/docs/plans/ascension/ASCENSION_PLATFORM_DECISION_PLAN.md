# Ascension Platform Decision Plan

**Status:** PLAN ONLY — no live Qwen/Gravity/CUDA work  
**Gate:** Proto-Frankenstein offload complete before any production Qwen path  
**Bible:** `HAWKING_ASCENSION_BIBLE.md` §1 (Platform decision), §34 (CUDA future)  
**Owns:** Tier 1 / Tier 1B / Tier 2 backend boundary only  

**Does not own (companion docs):**

| Concern | Document |
|---------|----------|
| What changes Gravity packing / mechanism research | [`ASCENSION_GRAVITY_RESEARCH_REGISTRY.md`](./ASCENSION_GRAVITY_RESEARCH_REGISTRY.md) |
| 30B parity / Gravity ladder rungs | [`ASCENSION_30B_PARITY_LADDER_PLAN.md`](./ASCENSION_30B_PARITY_LADDER_PLAN.md) |
| Programme order, CUDA slot in schedule | [`ASCENSION_PROGRAM_OVERVIEW.md`](./ASCENSION_PROGRAM_OVERVIEW.md) |

Terminology alignment: this document uses the bible’s **Tier 1 / Tier 2 / future Tier 1B** names. The 30B plan’s `ParityClassification.NUMERIC_PARITY_V2_1_ONLY` and DSV4F’s `DeepSeekV4P4bParityClassification::NumericParityV21Only` are the **same honesty label** at different surfaces (scaffold enum vs sealed Metal receipt). The Gravity research registry’s **ADMIT_TO_KERNEL / ADMIT_TO_GRAVITY** decisions feed packing and kernels; they do **not** redefine the platform tiers.

---

## 1. Governing decision (bible §1)

Hawking is:

```text
Apple-first
Metal-dominant
architecture-portable
CUDA-deferred
```

| Tier | Role | When |
|------|------|------|
| **Tier 1** | Production: Apple silicon, Metal, fully optimized, parity-gated, performance-gated | **Now** — only production backend |
| **Tier 2** | Portable contracts: model semantics, Gravity representation, kernel grammar/IR, scheduler contract, parity/capability harness, receipt schema, backend-neutral runtime interfaces | **Now** — designed and partially embodied in DSV4F |
| **Future Tier 1B** | CUDA production backend | **Only after** Apple release **and** funded hardware; never delays Tier 1 |

### Share / do-not-share (bible, restated as engineering law)

Metal and a future CUDA backend **MAY share**:

```text
architecture semantics
Gravity tensor semantics
benchmark contracts
parity / capability suites
scheduler interfaces
receipt formats
```

They **do NOT need to share** (and must not be forced into LCD abstractions):

```text
kernel source
tile geometry
memory layout
graph implementation
cache policy
command topology
autotuning rules
```

**Hard rules:**

1. Do not delay Apple release for CUDA.  
2. Do not force Metal through lowest-common-denominator abstractions.  
3. Apple claims remain Apple-specific; CUDA must re-earn parity and performance on real CUDA hardware.  
4. No CUDA implementation work in this programme slice (bible §34 / overview).

---

## 2. Living proof: DSV4F already implements the split

Tonight’s DeepSeek-V4-Flash work is **100% Apple/Metal Tier 1** for execution (custom SIMDgroup / authority kernels, real dispatch counts, honest parity), while **artifact admission, topology, scheduler seam, and receipt honesty labels** are already **Tier 2–shaped** even though only Metal implements the device side.

That is not aspirational architecture — it is how the modules are already factored under `crates/hawking-core/src/gravity_deepseek_v4*.rs` plus sealed receipts under `receipts/`.

---

## 3. Tier 2 — already portable-contract-shaped (cite real files)

These modules document (and mostly enforce) **no Metal allocation, no Engine, no TPS**. A CUDA backend would **consume** them, not rewrite them.

### 3.1 Gravity representation / artifact admission

| Surface | File | What is backend-neutral |
|---------|------|-------------------------|
| Full-stream schema + reader | `crates/hawking-core/src/gravity_deepseek_v4.rs` | `FULL_STREAM_SCHEMA = "hawking.gravity.deepseek_v4.full_stream.v1"`; content-addressed chunk tree; pinned repo/revision; verified range reads; `NativeScalePairKind` (`Fp4E2M1fnX2`, `Fp8E4M3fn`) and scale-pair geometry |
| Runtime binding sidecar | `crates/hawking-core/src/gravity_deepseek_v4_runtime_binding.rs` | `DSV4F_RUNTIME_BINDING_SCHEMA`; identity seals; `DeepSeekV4RuntimeAbi`; `DeepSeekV4SourceTensorGrammar`; cache residency **policy ceilings** (not Metal heaps) |
| Source-hash data plane | `crates/hawking-core/src/gravity_deepseek_v4_runtime_spine.rs` | Layer topology, compression modes, control/expert projections, bounded native FP8/FP4 staging, embedding/norm/head row primitives |
| Verified host tensor cache | `crates/hawking-core/src/gravity_deepseek_v4_verified_tensor_cache.rs` | Authenticated host-side control/tile cache; explicit “upload source only after device residency receipt” — no Metal buffers |
| Expert bundle cache | `crates/hawking-core/src/gravity_deepseek_v4_expert_cache.rs` | Hot/cold expert keying and access records as source-native bundles |

**Gravity tensor semantics** live here: source-native dtypes, packed-K vs logical-K, E8M0 scale locality, fail-closed hash verification. Backend choice starts **after** verified host bytes exist.

### 3.2 Architecture / model semantics

| Surface | File | What is backend-neutral |
|---------|------|-------------------------|
| Per-layer plan catalog | `crates/hawking-core/src/gravity_deepseek_v4_layer_plan.rs` | `DeepSeekV4LayerDevicePlan`: compression, gate mode, capability flags, **honest refusals** (e.g. ratio-4/128 non-BOS not implemented); `DeepSeekV4MhcControlExpStrategy` label for receipts |
| Source anchors | `crates/hawking-core/src/gravity_deepseek_v4_layer_source_anchors.rs` | Layer-bound tensor names and modes from sealed stream |
| Ratio-0 attention **plan** | `crates/hawking-core/src/gravity_deepseek_v4_attention_device.rs` | `DeepSeekV4Ratio0AttentionDevicePlan`, tensor name map, growing-KV params / kernel **identity string** — plan surface only; comment states Metal encoding lives in L0/L1/P4B executors |
| Execution context scaffold | `crates/hawking-core/src/gravity_deepseek_v4_execution_context.rs` | mHC slots, KV layouts, control leases, command-graph **ledger** (accounting), no encoder |
| CPU oracles / host math | `gravity_deepseek_v4_layer0_{prefix,attention,moe,continuation,position1_ffn}.rs`, `gravity_deepseek_v4_act_quant.rs`, `gravity_deepseek_v4_p0_gate_calibration.rs`, `gravity_deepseek_v4_final_head.rs` (host path) | Exact-model semantics for parity authority; independent of GPU API |

### 3.3 Scheduler contract / backend-neutral runtime interface

| Surface | File | What is backend-neutral |
|---------|------|-------------------------|
| Layer preparation scheduler | `crates/hawking-core/src/gravity_deepseek_v4_layer_scheduler.rs` | `DeepSeekV4LayerPreparationStage` (11 source-staging steps), `DeepSeekV4NativeStage`, **`DeepSeekV4NativeStageSink` trait**, `DeepSeekV4NativeStageConsumption` with `actual_gpu_dispatches` / CB / waits / host handoff bytes defaulting to **zero until a sink reports real work** |
| P7 composition (source lease prep) | `crates/hawking-core/src/gravity_deepseek_v4_p7_composition.rs` | Implements `DeepSeekV4NativeStageSink` for **source lease preparation** without requiring Metal for the composition contract itself; device executor is a separate trait boundary |

This is the **exact seam** bible §1 means by “scheduler contract” and “backend-neutral runtime interfaces”:

```text
DeepSeekV4LayerPreparationScheduler
  → DeepSeekV4NativeStage (source-native payloads)
  → DeepSeekV4NativeStageSink::consume_native_stage
  → DeepSeekV4NativeStageConsumption (honest dispatch accounting)
```

Metal today: `DeepSeekV4P3aMetalStageSink` in `gravity_deepseek_v4_p3a_stage_sink.rs`.  
Future CUDA: a `CudaStageSink` (name illustrative) implementing the **same trait**, leaving scheduler + spine + reader untouched.

### 3.4 Parity / capability harness labels (shared vocabulary)

| Surface | File / receipt | Portable contract |
|---------|----------------|-------------------|
| Terminal honesty enum | `gravity_deepseek_v4_p4b_device.rs` → `DeepSeekV4P4bParityClassification::NumericParityV21Only` → `"NUMERIC_PARITY_V2_1_ONLY"` | Classification is **not** “Metal-only”; it is a claim strength. Metal (and later CUDA) both emit it until exact-storage is sealed |
| P7 predecessor chain | `gravity_deepseek_v4_p7_device.rs` → `DeepSeekV4P7P4BPredecessorParity` | Same `NumericParityV21Only` inheritance rule for composed paths |
| Numeric suite schema | Receipts: `"schema": "hawking.numeric_parity.v2_1"` | Bounds (cosine, rel L2, etc.) are backend-agnostic; only the **device under test** changes |
| 30B ladder generalization | `ASCENSION_30B_PARITY_LADDER_PLAN.md` | Maps DSV4F labels → `ParityClassification`, `PASS_FULL_STACK`, `claim_boundary`, `fallback=0`, real GPU dispatch gates — **shared methodology**, family-specific stages |

Promotion rules that any backend must obey (already practice on Metal, required on CUDA):

1. `fallback_count != 0` → never promote.  
2. GPU rung with `gpu_dispatches == 0` → reject (no fake “GPU pass”).  
3. Numeric pass without full residual chain → `NUMERIC_PARITY_V2_1_ONLY` only.  
4. Exact-storage / full stack requires an explicit sealed promotion.

### 3.5 Receipt schema (backend-neutral envelope + backend-local body)

Sealed DSV4F receipts already separate **shared honesty** from **Metal physical evidence**:

| Schema (portable family) | Example path | Portable fields | Metal-local body |
|--------------------------|--------------|-----------------|------------------|
| `hawking.gravity.deepseek_v4.p4b_position1_complete_attention_metal.v1` | `receipts/DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v1-reseal-darwin-dd.json` | `schema`, `status`, `claim_boundary`, `source` seals, `numeric_parity_v2_1`, discrete stage parity flags | `metal.*` (device name, CB/dispatch/wait counts, pipeline limits, threadgroup widths) |
| `hawking.gravity.deepseek_v4.multi_layer_gpu_forward_bos.v1` | `receipts/dsv4f_multi_layer_gpu_forward_bos_l0_l2_receipt.json` | `parity.classification = NUMERIC_PARITY_V2_1_ONLY`, `exact_storage: false`, artifact identity, stage list | `metal.metal_dispatches`, `command_buffers`, `fallback` |
| `hawking.gravity.deepseek_v4.learned_bias_route_metal.v1` | `receipts/dsv4f_learned_bias_route_metal_receipt.json` | parity classification + scope | Metal dispatch block |
| `hawking.gravity.deepseek_v4.fullseq_capture_master.v1` | `receipts/dsv4f_fullseq_capture_master_receipt.json` | corpus honesty, parity class, totals | dispatch totals from Metal capture |

**Contract pattern for future CUDA receipts:** keep the same outer honesty fields (`schema` version family, `status`, `claim_boundary`, `parity.classification`, artifact identity, numeric_parity v2.1). Replace the `metal` object with a parallel `cuda` (or generic `device`) object — **do not** require Metal field names on non-Metal runs; **do** require the same mandatory honesty keys (fallback, real dispatch count, claim boundary).

Ascension scaffold schemas (not yet DSV4F live) are already named backend-neutral in the 30B plan: `hawking.ascension.parity_ladder_receipt.v1`, etc.

---

## 4. Tier 1 — Metal-specific by necessity (cite real files)

These are production Apple paths. A CUDA backend **must not** be implemented by “if metal {…} else {…}” inside them; it gets **sibling modules** (or a parallel crate feature) behind the Tier 2 seams above.

### 4.1 Metal runtime / dispatch substrate

| Surface | Location | Why Tier 1 only |
|---------|----------|-----------------|
| Metal context, batches, timing, physical traces | `crates/hawking-core/src/metal/` (`mod.rs`, arenas, argbufs, …) | OS/GPU API, command buffers, signposts |
| Shader sources | `crates/hawking-core/shaders/*.metal` (`matmul.metal`, `quant.metal`, `moe.metal`, `deepseek_v4_p7.metal`, `deepseek_v4_mhc_control_exp.metal`, …) | Metal Shading Language; tile/SIMD group geometry; Darwin DD control domain helpers |

### 4.2 Device executors (caller-owned `MetalContext`)

| Module | Role |
|--------|------|
| `gravity_deepseek_v4_p3a_stage_sink.rs` | Metal `DeepSeekV4NativeStageSink` — binds scheduler stages to Metal uploads/dispatches |
| `gravity_deepseek_v4_p4b_device.rs` | Position-1 complete attention device graph; fixed dispatch counts; `MetalBatchTiming` |
| `gravity_deepseek_v4_p6_device.rs` | MoE / expert wave device graph |
| `gravity_deepseek_v4_p7_device.rs` | mHC FFN composition on same Metal queue as P4B predecessor |
| `gravity_deepseek_v4_layer1_attention_device.rs` | L1 BOS attention; validates same queue identity; real `dispatch_*` chain |
| `gravity_deepseek_v4_bos_layer_attention_device.rs` | BOS multi-layer attention specialization |
| `gravity_deepseek_v4_fullseq_attention_device.rs` | Full-sequence capture device path (large dispatch counts) |

Shared Metal-only properties (must stay free of LCD abstraction):

- threadgroup sizes / SIMDgroup widths (receipts record `thread_execution_width`, `max_total_threads_per_threadgroup`)  
- command buffer / encoder topology (e.g. multi-layer L0–L2: **276 dispatches / 26 CB / 26 waits** — research registry)  
- host intermediate handoff policy (`host_intermediate_handoff_bytes == 0` as a Tier 1 quality bar)  
- pipeline precompile before dispatch  
- Apple device strings (`Apple M3 Ultra`)

### 4.3 Kernel grammar identity (shared **name**, not shared **source**)

Tier 2 may name the production kernel grammar, e.g.:

```text
deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority
```

(`gravity_deepseek_v4_attention_device.rs` constant `DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL`.)

That **string** is a portable architecture/kernel-grammar identifier for receipts and plans. The **Metal implementation** of that grammar (MSL source, tile layout, cache policy) is Tier 1. A CUDA implementation of the **same semantic kernel** is Tier 1B source — new files, not shared `.metal`.

This matches Gravity research registry §5.12: pack for primary custom SIMDgroup/authority family on Apple; do not pack for LCD layouts that force token-time conversion.

### 4.4 Examples / probes that are Metal product surfaces

Under `crates/hawking-core/examples/`: `gravity_deepseek_v4_*_metal*`, `*_gpu_forward*`, `*_simdgroup*`, `p4b_position1_complete_attention_metal`, multi-layer BOS forward, fullseq capture, etc. These produce the sealed receipts; they are not Tier 2 contracts themselves.

---

## 5. Boundary diagram (what a CUDA backend implements)

```text
                    ┌─────────────────────────────────────────┐
  TIER 2            │ Full stream reader + seals              │
  (shared)          │ Runtime spine / binding / tensor cache  │
                    │ Layer plan + source anchors             │
                    │ Execution context + expert cache        │
                    │ LayerPreparationScheduler               │
                    │ NativeStageSink trait + Consumption     │
                    │ Parity labels + numeric_parity.v2_1     │
                    │ Receipt envelope (schema/status/claim)  │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
     ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
     │ TIER 1 Metal    │    │ (future) T1B    │    │ Host oracles    │
     │ P3aMetalSink    │    │ CudaStageSink   │    │ (CPU authority) │
     │ P4B/P6/P7 dev   │    │ CUDA graphs/    │    │ parity refs     │
     │ shaders/*.metal │    │ kernels/        │    │                 │
     │ metal/ runtime  │    │ memory layout   │    │                 │
     └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                        │
              └────────────┬───────────┘
                           ▼
              Same promotion gates:
              fallback=0, real dispatches,
              NumericParityV21Only until
              exact-storage sealed
```

**Interface boundary — CUDA implements without touching Tier 1 Metal code:**

| # | Contract | Existing hook | CUDA delivers |
|---|----------|---------------|---------------|
| 1 | Artifact open + verified tensors | `DeepSeekV4FullStreamReader` / verified cache | Consume only; no alternate weight grammar without Gravity re-seal |
| 2 | Topology + layer capability | `DeepSeekV4LayerDevicePlan` / catalog | Honor refusals; implement missing graphs as new work with honest refusals until ready |
| 3 | Stage schedule | `DeepSeekV4LayerPreparationScheduler` | Drive same stages; do not invent parallel stage vocabularies |
| 4 | Device consumption | `DeepSeekV4NativeStageSink` | New sink type; report true `DeepSeekV4NativeStageConsumption` |
| 5 | Operator graphs | Attention/MoE/mHC **semantic** plans | New CUDA kernels/modules; **not** edits to `*_device.rs` Metal paths |
| 6 | Parity | `NUMERIC_PARITY_V2_1_ONLY` + numeric suite | Re-run on CUDA device outputs; no inherited Apple pass |
| 7 | Receipts | schema + claim_boundary + parity + dispatch honesty | New schema names ok (`…_cuda.v1`) if envelope fields match; never claim Metal dispatches |
| 8 | Benchmarks | shared contracts / scoreboard cells | Independent BASE_TRUE_TPS (or withheld) on CUDA hardware |

**Explicit non-interfaces (do not share implementation):**

- `crates/hawking-core/src/metal/**`  
- `crates/hawking-core/shaders/**`  
- Metal buffer layouts, argument buffers, concurrent groups, PSO cache policy  
- Threadgroup / SIMDgroup autotune tables  
- Command topology (one CB per micro-stage vs fused graphs)

---

## 6. What “architecture-portable” means operationally

| Portable now (Tier 2) | Portable later (still Tier 2, family-specific) | Not portable (per backend Tier 1/1B) |
|----------------------|-----------------------------------------------|--------------------------------------|
| Content-addressed Gravity stream contracts | Qwen3-MoE organ packing decisions (registry) | Metal vs CUDA kernel source |
| Native quant pair kinds + scale locality | 30B parity ladder stages (30B plan) | Tile sizes, shared mem / threadgroup policy |
| Scheduler stage names + sink trait | TG gauntlet methodology | Graph capture (Metal vs CUDA Graphs) |
| Parity classification vocabulary | HCLI product test catalog | Cache residency **implementation** (policy ceilings are portable; heaps are not) |
| Receipt honesty envelope | Numeric parity v2.1 bounds | Device strings, pipeline limits, autotune |

The Gravity research registry decides **what must change packing vs runtime/kernel** before Qwen Gravity; this platform plan decides **which of those artifacts are shared vs reimplemented per GPU API**. Registry ADMIT_TO_GRAVITY items become Tier 2 artifact law; ADMIT_TO_KERNEL items become Tier 1 (Metal now) and later Tier 1B (CUDA) **separately**.

---

## 7. Deferred work (explicit non-goals of this document / session)

Per bible and task contract:

- No Qwen download, Gravity packing, or live model work.  
- No CUDA implementation, CUTLASS, Triton, or CUDA Graphs work.  
- No touching `lab/operators/frankenstein_*` or Frankenstein evidence.  
- No LCD “one kernel IR that compiles to Metal and CUDA” project — forbidden by §1.  
- No delaying Apple Tier 1 optimization for hypothetical CUDA share-out.

CUDA begins only as **funded post-Apple-release** Tier 1B (overview step 33 / bible §34).

---

## 8. Work sequence when Tier 1B is authorized (planning only)

1. Freeze Tier 2 contracts (reader schema, spine geometry, scheduler trait, parity enums, receipt envelope) with version bumps if needed — **without** Metal breakage.  
2. Add feature/module isolation: Metal remains default production; CUDA behind explicit cfg/feature and hardware gate.  
3. Implement `NativeStageSink` + minimal one-stage CUDA proof (e.g. single matvec) with **real** dispatch accounting and Numeric Parity V2.1.  
4. Climb the same parity ladder methodology as Metal/30B plan — independent evidence.  
5. Only then expand graphs (attention, MoE, multi-layer).  
6. Never relabel Apple receipts as multi-backend.

---

## 9. File ownership map (this programme)

| Path pattern | Tier |
|--------------|------|
| `gravity_deepseek_v4.rs`, `*_runtime_spine.rs`, `*_runtime_binding.rs`, `*_verified_tensor_cache.rs`, `*_expert_cache.rs`, `*_execution_context.rs`, `*_layer_plan.rs`, `*_layer_source_anchors.rs`, `*_layer_scheduler.rs`, `*_attention_device.rs` (plan), host oracles | **Tier 2** (portable contract / source plane) |
| `gravity_deepseek_v4_*_device.rs` (P4B/P6/P7/L1/BOS/fullseq), `*_p3a_stage_sink.rs`, Metal-facing `*_p7_device.rs` dispatch paths | **Tier 1** Metal |
| `crates/hawking-core/src/metal/**`, `shaders/**` | **Tier 1** Metal |
| `receipts/dsv4f_*`, P4B reseal JSON | **Tier 1 evidence** using **Tier 2 honesty labels** |
| Future `*_cuda*` modules / receipts | **Tier 1B** (not started) |
| `workspace/docs/plans/ascension/ASCENSION_PLATFORM_DECISION_PLAN.md` | This doc — platform boundary |

---

## 10. Confidence and honesty

| Claim | Confidence |
|-------|------------|
| Bible §1 Tier 1/2/1B is the governing platform law | **High** — text is explicit |
| DSV4F already factors portable source/scheduler vs Metal device | **High** — module docs and Metal ref counts match |
| `DeepSeekV4NativeStageSink` is the primary CUDA plug-in seam | **High** for staged path; some executors still call `MetalContext` directly (P4B/L1) and would need **parallel** device modules, not sink-only |
| Receipt envelopes already separate `parity`/`claim_boundary` from `metal` | **High** — sealed JSON inspected |
| Full multi-backend Engine abstraction exists today | **Low / false** — deliberately absent; do not invent one for CUDA |
| Qwen 30B will reuse this split | **Medium-high** — 30B plan explicitly reuses DSV4F methodology; Qwen modules not written yet |

---

## 11. One-line summary

**Tier 1 is Apple Metal production kernels and command topology; Tier 2 is the sealed Gravity source plane, layer/scheduler contracts, parity vocabulary, and receipt honesty already visible in DSV4F; future CUDA is a funded Tier 1B sibling behind `NativeStageSink` + new device modules that re-earns the same gates without touching Metal source or delaying Apple.**
