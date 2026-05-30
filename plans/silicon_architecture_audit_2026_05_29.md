> **SUPERSEDED IN PART (2026-05-30) by [throughput_bible_2026_05_30.md](throughput_bible_2026_05_30.md) — the canonical strategy.**
> This plan's **topic catalog and per-topic dead-front conclusions remain valid reference**, but its **framing is superseded**: the "36 ms/token gap is REAL and GPU-side" premise (§0, Topic 2) was a token-count measurement artifact — decode is **~85% GPU-busy / kernel-bound**, the gap is ~12–15%, so the **megakernel (Topic 2) and cross-layer co-issue (Topic 7) are deprioritized/dead**, and **cross-engine (Topic 5) was empirically killed** by the same-day silicon-builds session. The 63-tps ceiling is reframed as per-bitwidth (~66 at Q4_K_M, ~99 at 3-bit) and the 26.6-tps anchor is open (Bible uses ~39 clean).
> Full mapping (every topic ↔ silicon # ↔ Bible axis): [throughput_bible_reconciliation_2026_05_30.md](throughput_bible_reconciliation_2026_05_30.md).

# Silicon-Architecture Strengthening Audit + Implementation Plan
**Date:** 2026-05-29
**Authoring session context:** decode at 26.6 dec_tps (predec default-on after 6f0209e); composition_decision_matrix complete; code-level dispatch overhead candidates (ICB, concurrent QKV, PSO, AMX-GEMV, CPU encode) all individually ruled out at sub-1% e2e ceilings.
**Target reader:** next session — execute AUDIT phase first, PRESENT findings, then IMPLEMENT only items that survive audit. Do not skip audit. Several claims below are silicon-architecture inferences that need empirical confirmation before any code changes.

---

## 0. Why this plan exists

Every code-level "host overhead" lever has been individually ruled out (see `decode_gap_anatomy_2026_05_24`, `gpu_us_accuracy_verified_2026_05_24`, `icb_production_scale_2026_05_24`, `qkv_concurrent_2026_05_24`, `pso_transition_dead_2026_05_24`). The 36 ms/token "gap" between sum(gpu_us) and decode wall is REAL and GPU-side, and the conclusion was: code-only investigation exhausted, only architectural levers remain.

This plan enumerates those architectural levers at the **silicon level**: M3 Pro hardware features, physical bandwidth ceilings, command-processor microarchitecture, multi-engine SoC topology, and cache-hierarchy custom layouts. Each topic is framed as **claim → silicon basis → current dismantle state → audit steps → expected findings → implementation if validated → ceiling → risks/interactions**, so the next session can decide GO / HOLD / DEAD per topic before writing kernel code.

The hard ceiling is **63 dec_tps** (LPDDR5 bandwidth wall for Qwen-3B-Q4_K_M unbatched). We are at 26.6 today = **42% of physical ceiling**. The 2.4× headroom is what this plan attacks.

---

## 1. M3 Pro silicon facts (reference table)

These facts are load-bearing for the audits below. Re-verify any that drive a GO/DEAD decision.

| Component | Spec |
|---|---|
| CPU | 11-core (5P+6E) or 12-core (6P+6E) @ ~4.05 GHz P / 2.75 GHz E |
| GPU | 14-core or 18-core, ~1.4 GHz, ~5.4 TFLOPS FP32 / ~10.8 TFLOPS FP16 |
| ANE | 16 cores, 18 TOPS int8 (Apple-reported) |
| AMX | 2 blocks (1 per P-cluster), 1024-bit SIMD, ~2 TFLOPS f16 GEMM peak per block |
| NEON | 128-bit per CPU core, ~50 GFLOPS f16/core sustained |
| LPDDR5 | 150 GB/s peak, ~120 GB/s usable (~80% efficiency) |
| GPU L1 cache | ~32-64 KB per core (Apple-undisclosed exact size) |
| GPU L2 cache | ~8 MB shared across GPU cores |
| System Level Cache (SLC) | ~24 MB shared SoC-wide (CPU/GPU/ANE all see it) |
| Cache line size | 128 bytes (L1/L2 GPU) |
| Page size (default) | 16 KB (Apple Silicon default) |
| Super-page available | 2 MB via VM_FLAGS_SUPERPAGE_SIZE_2MB |
| TLB entries | ~64-128 L1 / ~1K-4K L2 (typical; Apple-undisclosed) |
| GPU command processor | Single-threaded HW frontend per command queue |
| Process node | TSMC N3B (3nm class) |
| Dynamic Caching | NEW on M3+: registers + L1 cache merged into single dynamically-allocated pool |

**Sources to verify against:** Apple Metal Feature Set docs, WWDC 2023 "Discover Metal 3" + "Meet the M3 family" sessions, hexops/`MetalBench` if installed, third-party teardowns (AnandTech M3 Max).

---

## 2. How to execute this plan

**Phase A — AUDIT (no code changes):**
For each topic, run the audit steps in order. Record measurements in `reports/silicon_audit/<topic_name>/`. Produce a per-topic verdict: **GO** (audit confirms ceiling justifies implementation), **HOLD** (results unclear, needs more data), **DEAD** (audit kills the hypothesis). Compile all findings into `reports/silicon_audit/_findings.md`.

**Phase B — PRESENT:**
Present the findings markdown to the user before any implementation. Highlight: top 2 GO items, dead items with one-line cause-of-death, surprising findings, and recommended ordering.

**Phase C — IMPLEMENT:**
Only after user approval, implement GO items per the prescribed sub-steps. Each item ships as a single commit per the CLAUDE.md single-purpose-commit rule. Bench with `tools/bench/paired_bench.sh` (or current equivalent) at n≥5 trials with `--strict`. Parity gate atol=1e-3 fp16 + bit-identical 3-tok per the project verification rule.

Halt rules per `CLAUDE.md`: G1.2-style 2-halt budget for kernel/runtime topics. Per-item soft ceiling 60 min; haul hard ceiling 4 hr.

---

## 3. Bench infrastructure to use during audit

| Tool | Purpose |
|---|---|
| `tools/bench/paired_bench.sh` | Standard paired A/B bench for e2e tps |
| `cargo test -p dismantle-kernels --test phase1_kernel_parity` | Parity validation atol=1e-3 |
| `cargo run -p dismantle --release -- bench --prompt-len 16/64/256` | Decode-only at fixed prompt length |
| `xcrun xctrace record --template "Metal System Trace"` | Instruments capture for command-processor + GPU activity |
| `xcrun metal --emit-llvm` / `metal-source-dump` | Kernel register count + occupancy report |
| `vmmap <pid>` | Page layout of weight buffers |
| `sysctl hw.l1dcachesize hw.l2cachesize` | Verify cache sizes on test machine |
| `powermetrics --samplers gpu_power,smc -i 1000` | GPU utilization + thermal during decode |

Instruments-required topics (cannot be answered from code-only): **2 (command-processor decomp), 3 (TLB walks), 4 (L2 hit rate), 6 (occupancy)**. If Instruments access is blocked, those audits halt with `reason: instruments_unavailable` and the items stay HOLD.

---

## 4. Audit + Implementation Topics

Eight topics. Read all before starting Phase A so you can plan instrument capture across multiple topics in a single trace (saves time vs re-capture per topic).

---

### Topic 1: LPDDR5 Memory Bandwidth Ceiling (the reference point)

**Claim:** Physical decode ceiling for Qwen-3B-Q4_K_M unbatched on M3 Pro is ~63 dec_tps. We are at 42% of ceiling. All other topics push toward this wall.

**Silicon basis:**
- LPDDR5-6400 quad-channel: 150 GB/s peak.
- Real-world sustained: ~80% efficiency = 120 GB/s.
- Qwen-3B-Q4_K_M on disk: ~1.93 GB (verify: `du -h <gguf path>`).
- Per-token decode reads ≈ full weight set + KV cache delta + activations.
- Minimum decode time = 1.93 GB / 120 GB/s = **16.1 ms/token = 62.1 dec_tps**.

**Current dismantle state:**
- 26.6 dec_tps median = 37.6 ms/token wall.
- Computed bandwidth: 1.93 GB / 37.6 ms = 51.3 GB/s = 34% of physical ceiling.
- KV cache + activation bandwidth not yet measured.

**Audit steps:**
1. Capture Instruments MST during a 32-token decode. Read GPU memory bandwidth gauge (peak + sustained). Record `actual_GB_per_sec.json`.
2. Compute theoretical per-token bandwidth: parse weight sizes from `qwen_dense.rs` arena layout + KV cache size at 256 ctx + activation footprint per layer × 28 layers. Record as `theoretical_bytes_per_token.json`.
3. Sanity check: actual_GB_per_sec ≈ (theoretical_bytes_per_token × dec_tps). If ratio ≠ 1.0 ± 0.15, something is mis-modeled — debug before proceeding.
4. Verify GGUF file size and weight layout match assumption (some Q4_K weights are stored deduped or compressed at rest).
5. Cross-check at multiple prompt lengths (16, 64, 256 tok) — bandwidth should scale near-linearly with kv-cache size if KV reads are significant.

**Expected findings:**
- **GO (ceiling correct):** Actual sustained = 50-65 GB/s; ceiling at 120 GB/s; ~2× headroom. → All other topics greenlit to pursue.
- **DEAD (ceiling already hit):** Actual sustained > 100 GB/s; we are already 80% of ceiling. → Only multi-engine concurrency (Topic 5) and dispatch-reduction (Topic 2) matter; abandon cache/layout topics.
- **SURPRISE (unexpected bandwidth target):** Activations or KV cache dominates over weights. → Focus topics 4 and 8 on those, not Q4_K weights.

**Implementation if validated:** None directly — Topic 1 is the *reference point*. All other topics measure themselves against this ceiling.

**Ceiling estimate:** N/A (it IS the ceiling).

**Risks / interactions:** If sustained bandwidth measurement is impossible without Instruments, fall back to computed model + per-kernel `gpu_us` sums. Record measurement methodology in finding.

**Effort:** 2-3 hours (mostly Instruments capture + analysis).

---

### Topic 2: GPU Command-Processor Serialization (~147 µs/dispatch gap)

**Claim:** The 36 ms/token gap between sum(gpu_us) and decode wall is GPU command-processor frontend serialization, ~441K GPU cycles per kernel-to-kernel transition. Collapsing dispatch count (megakernel) is the only way to eliminate it.

**Silicon basis:**
- M3 GPU command processor is a single HW frontend per command queue. Per Apple Metal docs (Metal 3 ICB notes), it processes one descriptor at a time.
- Per-dispatch frontend work:
  - Parse argument-buffer descriptor (~10K cycles)
  - Check buffer residency / page table install (~5K cycles + TLB walk cost)
  - Install PSO (pipeline state object) — ~1K cycles if cached, ~10K+ if cold
  - Spawn wavefronts to compute units (~5K cycles, depends on grid size)
  - Drain fence at completion (~5K cycles min)
- At 3 GHz GPU clock, 147 µs = **441K cycles** — consistent with the sum above + cache cold-start.
- Cache cold-start: new kernel's tile data typically not in L2 → first wavefront stalls on LPDDR fetch (~150 ns × multiple lines = 1-5 µs additional).

**Current dismantle state:**
- 235 dispatches/token at decode (per `decode_gap_anatomy_2026_05_24`).
- 34.6 ms/token gap = 96% of decode wall - sum(gpu_us).
- ICB POC: cut per-dispatch *CPU encode* from 4.19 → 0.99 µs (76% reduction) but only +0.9% e2e because CPU encode was only 3.72% of gap. So gap is GPU-side, not CPU-side.
- Megakernel: skeleton at day 3 (stage A only), pass-through invariant green, kernel body not implemented. Production-ready functional megakernel is 3-5 sessions out per `build_megakernel_day3_2026_05_25`.

**Audit steps:**
1. **Instruments MST capture during single-token decode.** Filter to GPU command processor timeline. Identify: a) frontend active periods, b) compute-unit active periods, c) idle gaps. Quantify each in µs/dispatch.
2. **Decompose the 147 µs gap into components:**
   - Frontend parse + dispatch setup: measure as time from "previous compute end" to "next compute start" minus any explicit fence wait.
   - PSO switches: count unique PSO IDs across the 235 dispatches → estimate switch cost as gap × frequency.
   - Cache cold-start: identify gap *within* a kernel's first wavefront execution (warm-up tail).
   - Hidden barriers (e.g., implicit memory barriers between buffer reads + writes): identify via MTLEvent/barrier markers in trace.
3. **Cross-validate via dispatch-count reduction microbench:** synthetic kernel that does N×1µs work as either (a) 1 dispatch with N iterations, (b) N dispatches with 1 iter each. Measure gap = (b_wall - a_wall) / N. This isolates pure frontend cost.
4. **Test ICB on the entire decode pass:** wrap all 235 dispatches into a single MTLIndirectCommandBuffer (Metal 3.2 supports this for compute). Measure delta. ICB POC reduced *CPU encode*, but Metal 3.2 ICB may also reduce GPU-side frontend by pre-resolving residency.

**Expected findings:**
- **Strong GO (frontend dominates):** Gap decomposes as ≥60% frontend serialization, ≤20% cache cold-start. → Megakernel is the right lever, eliminates all of it. Lean harder into megakernel workstream.
- **Mixed GO (frontend + cache):** Gap is 40% frontend + 40% cache cold-start + 20% other. → Megakernel + heap residency (Topic 4) must combine to win.
- **DEAD (frontend small):** Gap is mostly hidden barriers or implicit memory ordering. → Megakernel ROI smaller than projected; pivot to multi-engine (Topic 5).
- **SURPRISE:** ICB-on-decode-pass closes most of the gap. → Megakernel project descopable; ICB+predec stack becomes default.

**Implementation if validated (assuming strong GO):**
- Continue megakernel workstream per existing plan (`build_megakernel_day3_2026_05_25` is at stage A; needs stages B-L).
- Alternative path: build single-pass ICB harness for entire decode. ~3-5 days standalone scope.
- Megakernel projection: 235 → 2 dispatches/layer × 28 layers = 56 dispatches → saves ~179 × 147 µs = 26.3 ms/token.
- ICB-decode projection: 235 → 1 ICB execution → saves 234 × 147 µs = 34.4 ms/token IF GPU-side frontend collapses too (not just CPU encode).

**Ceiling estimate:** If gap fully collapses, decode wall drops to ~10 ms/token = **100 dec_tps**, capped by Topic 1 LPDDR wall at 63 dec_tps. Net: 26.6 → ~63 dec_tps = **2.4× e2e**.

**Risks / interactions:**
- Megakernel multi-week implementation risk per existing tracking memo.
- ICB-decode path is untested at full forward-pass scale (POC was microbench).
- Both depend on Topic 4 heap residency for cache cold-start; do not implement in isolation if cache turns out to dominate gap.

**Effort:** Audit 4-6 hours (MST capture + decomposition + ICB sweep). Implementation 3-15 days depending on path chosen.

---

### Topic 3: TLB / Page Walks Per Dispatch

**Claim:** 235 dispatches × ~26 buffer bindings = ~6,100 buffer touches per token. Weights spanning 1.93 GB at 16 KB pages = ~126K pages. TLB pressure compounds to ms-scale per token; super-page allocation can eliminate it.

**Silicon basis:**
- M3 default page size: 16 KB.
- TLB entries (Apple-undisclosed; Cortex-class typical): ~64-128 L1, ~1K-4K L2.
- TLB miss → page table walk: 4-level walk × ~3-5 ns per level = ~15-25 ns per miss.
- Worst case (cold weights, scattered VM): 6,100 buffer touches × 25 ns = ~150 µs/token. Small but real.
- Super-pages (2 MB): reduce page count by 128×. 126K pages → ~1K super-pages. Fits in TLB L2 → near-zero misses on hot data.
- Allocation API: `vm_allocate` with `VM_FLAGS_SUPERPAGE_SIZE_2MB` (Mach trap), or `madvise(MADV_HUGEPAGE)` not supported on Darwin.

**Current dismantle state:**
- No instrumentation of TLB misses.
- `load_heap_resident` POC committed (8acf069) consolidates weights into single MTLHeap → fewer VM regions → likely fewer TLB entries needed.
- **REVERTED at -5.3% e2e** because per-buffer `useHeap:` encoder calls per-dispatch were expensive. Migration cost did not pay back at steady-state decode. → Held infra unused.

**Audit steps:**
1. **vmmap inspection:** `vmmap <pid>` during `dismantle generate` → identify weight buffer VM layout. Count distinct VM regions, check page-size hints.
2. **TLB miss measurement (if possible):**
   - Try `instruments -t "System Trace"` → look for TLB miss events. Apple may or may not expose this for GPU.
   - Fallback: synthetic CPU benchmark — read 1.93 GB in 16 KB pages (sequential vs random) vs same in 2 MB super-pages. Approximates GPU pressure proportionally.
3. **2 MB super-page allocation test:**
   - Allocate 2 GB region via `vm_allocate` + `VM_FLAGS_SUPERPAGE_SIZE_2MB`.
   - Copy Qwen-3B weights into it.
   - Wire into `qwen_dense.rs` arena loading behind `DISMANTLE_QWEN_SUPERPAGE=1`.
   - Bench paired vs baseline.
4. **Heap residency revisit (critical):** the previous POC revert was due to per-dispatch `useHeap:` cost. The fix: batch `useResources:[heap.resources]:` once per encoder, not per-dispatch. Re-bench with this correction.

**Expected findings:**
- **GO (super-pages win):** Super-page allocation delivers measurable wall-clock reduction (≥3%) at decode. TLB walks were silently costing us ms-scale.
- **GO (heap-residency v2 win):** Batched-useResources heap path delivers ≥3% e2e, unlike the per-dispatch v1 path.
- **DEAD:** Super-page wall delta < 1%; weights already mostly TLB-resident due to VM coalescing. Drop topic.
- **SURPRISE:** Allocation fails (insufficient contiguous 2 MB regions); fall back to MTLHeap with placement strategy.

**Implementation if validated:**
- Add `arena.rs::alloc_superpage_region(size: usize) -> Result<NonNull<u8>>` helper using `mach_vm_allocate` FFI.
- Wire into Qwen weight loading: single super-page-backed region, all weights placed inside.
- Heap residency v2: `MTLHeap` with all weights, batched `enc.use_resources(&heap.allocated_resources(), .Read)` once per command buffer.
- Bench paired n≥5.

**Ceiling estimate:** If TLB walks are 100-200 µs/token, this is ~0.5% e2e — small. If TLB walks are 2-5 ms/token (worst case scattered VM), this is +5-13% e2e. Audit step 1 will tell us.

**Risks / interactions:**
- Super-page allocation may fail under memory pressure (especially with slm co-running per `CLAUDE.md` Memory-coexist rule). Need fallback to 16 KB pages.
- Heap residency v2 may conflict with megakernel argbuf strategy. Plan integration before both ship.
- TLB pressure may be dominated by KV cache, not weights — re-target accordingly.

**Effort:** Audit 3 hours. Implementation 4-8 hours.

---

### Topic 4: L2 / SLC Cache Residency for Hot Rows

**Claim:** LM-head and KV cache hot data currently cold-miss to LPDDR on every reread. M3 Pro has ~8 MB GPU L2 + ~24 MB SLC. Strategic placement of hot rows in these tiers can eliminate ms-scale LPDDR roundtrips.

**Silicon basis:**
- GPU L2: ~8 MB shared across cores. Latency ~5-10 ns. Auto-managed (LRU + hint-driven).
- SLC: ~24 MB shared SoC-wide (CPU, GPU, ANE all see it). Latency ~20 ns.
- LPDDR5: ~80-150 ns. Bandwidth 120 GB/s usable.
- Apple residency control APIs: `MTLResource.setPurgeableState(.nonVolatile)`, `MTLResidencySet` (Metal 3.2), `MTLHeap` placement hints.
- SLC has no direct control — but allocation patterns (page locality, access frequency) bias eviction.

**Current dismantle state:**
- LM head: ~vocab_size × hidden = 152K × 2048 × 0.5 bytes = ~152 MB at Q4_K. **Far exceeds L2.**
- vocab-prune-32K: reduces to 32K × 2048 × 0.5 = ~32 MB. **Still exceeds L2 (8 MB), but fits in SLC (24 MB) tightly.**
- KV cache at 256 ctx, fp16: 2 × 256 × 28 × 8 × 128 × 2 bytes = ~29 MB. **Just exceeds SLC.**
- KV cache at 256 ctx, Q8 (per `q8_kv_runtime_landed.md`): 14.7 MB. **Fits in SLC.**
- No L2/SLC residency controls in dismantle code today.

**Audit steps:**
1. **GPU L2 hit-rate profiling:** Instruments GPU Memory counters during LM-head GEMV. Record L1/L2/system hit rates. Repeat for: full LM head, vocab-pruned 32K, hypothetical top-1024.
2. **KV cache pressure measurement:** at 256 / 512 / 1024 ctx lengths, measure attention kernel time. If scaling is super-linear with ctx, KV is overflowing L2/SLC.
3. **Q8 KV residency test:** with Q8 KV (existing `--q8-kv` flag), re-measure L2 hit rate vs fp16. Expect Q8 to fit better.
4. **Top-K LM head warm-cache test:**
   - Pre-sample 100 prompts, identify top-1024 most-frequent vocab tokens.
   - Build a dedicated 1024-row sidecar buffer (1024 × 2048 × 0.5 = 1 MB).
   - On decode, check sidecar first, fall through to full LM head.
   - Bench paired.
5. **MTLResource.setPurgeableState test:** allocate KV cache as `.nonVolatile`. Measure if L2/SLC eviction behavior changes (very hard to measure directly; proxy via decode wall delta).

**Expected findings:**
- **GO (L2 hit rate low):** Full LM head L2 hit rate < 60%; vocab-pruned still < 80%; top-K sidecar drives L1+L2 > 95%. → Implement top-K residency.
- **GO (SLC matters):** KV cache scaling super-linear with ctx; Q8 KV closes gap. → Make Q8 KV default-on for sufficient ctx.
- **DEAD:** L2/SLC hit rates already > 90% across configurations. → No headroom; drop topic.
- **SURPRISE:** Attention is L2-thrashed more than LM head. → Pivot to KV cache layout (Topic 8 extension).

**Implementation if validated:**
- **Top-K LM head sidecar:**
  - Add `tools/calibrate_topk_vocab.py` — sample N prompts (use `corpus_complete_analysis_landed` corpus), produce ranked vocab.txt.
  - At Qwen load time, build a 1024-row sidecar buffer from the top-K rows of the full LM head.
  - At decode logits step, GEMV against sidecar → if argmax ∈ sidecar, return; else fall through to full LM head GEMV.
  - Behind `DISMANTLE_QWEN_TOPK_LMHEAD=1`.
- **SLC-friendly heap placement:**
  - Allocate KV cache via dedicated `MTLHeap` with `MTLHeapType::Placement`.
  - Mark as `.nonVolatile` via `setPurgeableState`.
  - Re-bench at multiple ctx lengths.
- **Q8 KV default-on:** if SLC residency proves load-bearing for KV, flip `DISMANTLE_QWEN_Q8KV=1` to default.

**Ceiling estimate:**
- Top-K LM head: if 95% of decoded tokens are in top-1024, saves 95% × LM head time (~4% of decode) × LPDDR-to-L2 latency factor (~10×) = **~3% e2e**.
- SLC KV: at long ctx, saves attention LPDDR roundtrips = +5-10% e2e at 1024+ ctx.
- Combined: +5-15% e2e at typical workloads.

**Risks / interactions:**
- Top-K sidecar quality risk: if argmax falls outside top-K, we double LM-head work (sidecar + full). Need to gate by token-rank confidence or use only as prefetch.
- SLC has no direct control; effects are second-order and noisy. May need n≥20 trials.
- `setPurgeableState(.nonVolatile)` may force LPDDR pin, not L2 pin — verify semantics.
- Conflicts with Topic 3 super-page allocation if both consume contiguous VM.

**Effort:** Audit 4-6 hours (Instruments-heavy). Implementation 1-2 days for top-K sidecar; ~4 hours for SLC heap.

---

### Topic 5: Cross-Engine Concurrency (ANE / AMX / NEON + GPU)

**Claim:** M3 Pro has 4 independent compute engines (GPU, ANE, AMX, NEON). dismantle uses only GPU. Concurrent decode across engines could double effective compute throughput — IF they have separate bandwidth paths to LPDDR. This is the highest-ceiling unaudited lever in the project.

**Silicon basis:**
- **GPU (14-18 cores):** 5.4 TFLOPS FP32 / 10.8 TFLOPS FP16. Bandwidth-bound at decode (~51 GB/s of 120 GB/s).
- **ANE (16 cores):** 18 TOPS int8. *Critical unknown:* private DMA path or shared LPDDR bus? Apple docs are opaque. Some sources suggest ANE has dedicated bandwidth (separate memory controller channel); others suggest shared.
- **AMX (2 blocks):** 2 TFLOPS f16 GEMM peak per block. Shared by P-cores (one block per P-cluster). Accessed via undocumented instructions or Accelerate framework (cblas, BNNS).
- **NEON (per P-core):** 50 GFLOPS f16. Mostly idle during GPU decode.

Cache topology:
- GPU has private L1/L2.
- ANE has private SRAM (unspecified size, est. 8-16 MB on M3).
- AMX has small staging buffers (~16 KB per block).
- NEON shares CPU L1d (16 KB per core).
- SLC is shared by all (24 MB).

**Current dismantle state:**
- AMX ruled DEAD at GEMV (`amx_feasibility_2026_05_24`): 50 GFLOPS sustained, 1.1-2.5× SLOWER than GPU at GEMV shapes. But that test was GEMV-only — never tried at GEMM.
- ANE: never tested in dismantle. No prior art in codebase.
- NEON: used in some Q4_K dequant CPU paths, not decode hot path.

**Audit steps:**

**5a. ANE bandwidth topology probe (the critical unknown):**
1. Build standalone Rust crate `crates/ane-bandwidth-probe/` (sidesteps main build state).
2. Implement: ANE memcpy 1 GB + simultaneous GPU memcpy 1 GB. Time both.
3. If ANE + GPU concurrent time = max(ANE_alone, GPU_alone) → **private bandwidth** → high-value GO.
4. If concurrent time ≈ ANE_alone + GPU_alone → **shared bandwidth** → ANE only useful for compute-bound, not bandwidth-bound (= most decode).
5. Repeat with mixed read/write patterns to test cache coherency cost.

**5b. AMX GEMM (re-evaluate at batch sizes that weren't tested):**
1. Re-run `amx-spike` (`claude/amx-spike-2026-05-24` branch) at:
   - B=1 (GEMV — known DEAD)
   - B=4 (Eagle5 draft batch size)
   - B=8, B=16, B=32 (batched serving)
2. Compare TFLOPS sustained vs GPU at each B.
3. AMX may win at B≥8 where GPU underfills compute units.

**5c. ANE CoreML inference benchmark:**
1. Convert one Qwen FFN layer to CoreML mlpackage via `coremltools` Python.
   - Input: residual (f16 or f32, shape `[1, hidden]`).
   - Output: post-FFN residual.
   - Weights: ffn_gate, ffn_up, ffn_down (Q4_K → CoreML may require fp16 conversion; record quality delta).
2. Build Rust harness via `apple-ml/coreml-rs` or direct objc bindings.
3. Measure single-FFN latency on ANE. Compare to GPU FFN latency.
4. If ANE FFN < GPU FFN: GO.
5. If ANE FFN ≥ GPU FFN but bandwidth is private (per 5a): GO (concurrent execution wins anyway).

**5d. Concurrent decode pipeline test:**
1. Stage GPU + ANE: GPU runs attention(layer N), ANE runs FFN(layer N-1) speculatively or prep.
2. Synchronization: MTLSharedEvent + Core ML completion handler.
3. Measure paired n=10 vs serial baseline.

**5e. NEON+AMX pre/post processing offload:**
1. Move softmax (currently GPU) to NEON.
2. Move RoPE (currently GPU) to AMX.
3. Measure GPU-side savings + CPU-side cost.
4. Goal: free GPU to spend 100% time on GEMM, push closer to LPDDR ceiling.

**Expected findings:**
- **Highest-ceiling GO (private ANE bandwidth):** ANE memcpy concurrent with GPU memcpy hits 1.6-2× combined throughput → ANE has private bandwidth → ANE-FFN + GPU-attn pipeline projects 1.5-2× e2e. **Single biggest lever in this plan.**
- **Mid GO (AMX wins at GEMM):** AMX at B=8 hits >2× GPU GEMM TFLOPS → adopt for Eagle5 draft batch + batched serving.
- **Small GO (offload only):** NEON softmax + AMX RoPE recovers 3-5% GPU time → marginal but cheap.
- **DEAD:** ANE shares bandwidth + CoreML conversion lossy + AMX no win at any B. → Drop entire topic except possibly NEON softmax.
- **SURPRISE:** ANE int8 quant of Qwen FFN matches GPU Q4_K quality and runs faster → blueprint for ANE-native LLM serving on Apple Silicon, paper-worthy result.

**Implementation if validated (best case):**
- **Phase A: standalone crates:**
  - `crates/ane-spike/` — ANE FFN benchmark + correctness validator.
  - Already exists: `crates/amx-spike/`.
- **Phase B: CoreML model artifacts:**
  - `tools/convert_qwen_ffn_to_coreml.py` — produce `qwen_ffn_layer.mlpackage`.
  - Place at `_release_assets/coreml/qwen3b_ffn.mlpackage`.
- **Phase C: Rust harness:**
  - `crates/dismantle-coreml/` — load mlpackage, run prediction, synchronize with MTLSharedEvent.
- **Phase D: integration:**
  - `qwen_dense.rs` decode loop adds optional ANE-FFN path behind `DISMANTLE_QWEN_ANE_FFN=1`.
  - GPU runs attention while ANE runs prior layer's FFN.
- **Phase E: NEON softmax + AMX RoPE offload (if validated):**
  - Add `crates/dismantle-cpu-offload/` for NEON/AMX kernels.
  - Wire behind `DISMANTLE_QWEN_OFFLOAD_SOFTMAX=1` and `DISMANTLE_QWEN_OFFLOAD_ROPE=1`.

**Ceiling estimate:**
- ANE private bandwidth case: 1.6-2× e2e → 26.6 → 42-53 dec_tps. Approaches LPDDR ceiling.
- ANE shared bandwidth case: +20% from compute parallelism alone → 26.6 → 32 dec_tps.
- AMX GEMM case: +30-50% for batched serving paths; minimal for unbatched decode.
- NEON+AMX offload: +3-8% e2e.

**Risks / interactions:**
- ANE-FFN quality risk: Q4_K → fp16 CoreML conversion may degrade. Needs corpus quality gate (similar to W4A8 quality sweep).
- ANE latency floor: ANE has high per-inference overhead (~ms-scale per launch) — may be too slow for streaming decode at the per-token level. Need to amortize across layers or use as background prep.
- CoreML model artifact ships at ~100s of MB — repo bloat concern.
- AMX is undocumented; relies on Accelerate.framework which may change between macOS versions.
- Topic 5 is by far the **highest-risk-highest-reward** topic. Allocate 1-2 weeks for full audit + spike before committing to implementation.

**Effort:** Audit 5-7 days (multiple crates, CoreML conversion, instrumentation). Implementation 2-4 weeks if validated.

---

### Topic 6: M3 Dynamic Caching Exploitation

**Claim:** M3 family GPU merged register file and L1 cache into a single dynamically-allocated pool. Current kernels written for pre-M3 fixed-register-budget assumptions miss this. Fat threadgroups + larger tiles could 1.2-2× kernel throughput on M3+.

**Silicon basis:**
- Pre-M3 Apple GPU: fixed register file (~16-32 KB per simdgroup). Kernels exceeding budget spill to threadgroup memory or system memory at high latency.
- M3 GPU (per WWDC 2023 "Discover Metal 3" + Apple chip presentation):
  - Register file + L1 cache merged into single unified pool.
  - Allocation dynamically rebalanced per active kernel.
  - Net: register-pressure kernels get more registers without spill; cache-pressure kernels get more cache.
- Implication: kernels written conservatively for pre-M3 leave M3 perf on table.
- `MTLResidencySet` (Metal 3.2) API: explicit resource residency control, may interact with Dynamic Caching.

**Current dismantle state:**
- Most Q4_K kernels use 256-thread threadgroups (typical Apple GPU sweet spot for pre-M3).
- No M3-specific variants exist.
- Per `path_to_30_findings.md`: MoE v3 kernel suffered -14% from register pressure on pre-M3 architectures. This may *reverse* on M3 with Dynamic Caching.
- Predec already uses pre-decoded tile data which fits more in registers — partial M3 benefit incidentally.

**Audit steps:**
1. **Register report:** `xcrun metal --emit-llvm` on key Q4_K kernels → extract register count + occupancy estimate per simdgroup.
2. **Occupancy measurement:** Instruments GPU Counters → measure active simdgroups per core during Q4_K GEMV. Compare to theoretical max.
3. **Detect M3 at runtime:** Metal device family check (`MTLGPUFamily::Apple9` or via `device.supportsFeatureSet`). Confirm we're on M3 hardware.
4. **Fat threadgroup variant:** rewrite `gemv_q4_k_v2t_gu_v2` with 1024 threads + larger per-thread tile. Compare on M3 Pro:
   - Kernel time
   - Register usage (compile-time report)
   - Occupancy (runtime)
5. **Tile-reuse variant:** keep one layer's full weight tile in L1 across Q/K/V/O sub-GEMVs. Test if M3 cache hint enables this.

**Expected findings:**
- **GO (fat tg wins on M3):** 1024-thread variant 1.3-1.8× faster on M3. → Add `_m3` variants, runtime-dispatch.
- **DEAD (no benefit):** Fat tg same or slower; register file already optimally sized. → Drop.
- **MIXED:** Some kernels benefit (tile-reuse-heavy), others don't (streaming-heavy). → Selective adoption.
- **SURPRISE:** Dynamic Caching reverses MoE register-pressure regression. → Revisit MoE v3.

**Implementation if validated:**
- Add `_m3` kernel variants in `kernels/q4_k/v3_m3.metal` (or similar).
- Runtime dispatcher in `kernels/mod.rs` detects Apple9+ family → routes to `_m3` variants.
- Parity gate: atol=1e-3 + bit-identical 3-tok.
- Bench paired n≥5 on M3 Pro target hardware.
- Document fallback for pre-M3 hardware (M1/M2).

**Ceiling estimate:** 1.2-1.5× on bandwidth-bound kernels. Most of forward pass is bandwidth-bound → +20-40% e2e if applied broadly. Compounds with Topic 8 layout improvements.

**Risks / interactions:**
- M3-only — does not help M1/M2 users. Need fallback path.
- Compile-time register report may not predict runtime occupancy accurately on M3 due to dynamic allocation.
- Compounds with Topic 8 (layout) and Topic 4 (cache hints) — order matters; tune together.

**Effort:** Audit 2-3 days. Implementation 1-2 weeks for full kernel sweep + benchmarking.

---

### Topic 7: Speculative Kernel Co-Issue Across Layers

**Claim:** Decode is fully serial across layers. Some next-layer ops (Q projection input is `rmsnorm(prev_residual)`) depend only on partial prior-layer state. Issuing these speculatively on a second command queue hides serial wait latency.

**Silicon basis:**
- M3 GPU command processor supports parallel kernel execution if resources don't conflict.
- Metal queue model: one command queue = serial. Multiple queues = parallel, synchronized via MTLEvent.
- NVIDIA equivalent: CUDA streams + CUDA Graphs.
- Apple lacks built-in graph DAG API but supports per-stage MTLSharedEvent fencing.

**Current dismantle state:**
- Single command queue.
- Layer N+1 entirely waits for layer N completion.
- Per `per_kernel_time_breakdown.md`: GEMM is 50.5% of decode; attention only 2.4%. Most decode time is GPU-active, not idle-waiting.
- gpu_us measurement confirms 70% of decode is gap (not GPU-active). But: is the gap *waiting* or *frontend-busy*? Topic 2 audit will tell us.

**Audit steps:**
1. **Cross-reference Topic 2 findings:** if the 147 µs gap is frontend-busy (command-processor parsing next descriptor), this topic is DEAD — no idle to fill. If gap is idle (waiting for fence drain), this topic has headroom.
2. **Build inter-layer dependency DAG:** map data flow across 2-3 layers. Identify earliest-startable layer N+1 ops.
3. **Speculative-issue prototype:**
   - Second command queue for "speculative" work.
   - Issue layer N+1 rmsnorm + Q-proj as soon as layer N residual is written (before final add finalizes — assumes commutativity).
   - MTLSharedEvent fence at layer boundary verifies dependency.
4. **Discard-on-mismatch test:** what fraction of speculative work is reusable? Should be ~100% for forward pass (deterministic), but measure.
5. **Wall-clock paired bench n≥10.**

**Expected findings:**
- **GO:** gap-was-idle confirmed in Topic 2 → speculative co-issue recovers 5-15% e2e.
- **DEAD (Topic 2 says gap is frontend-busy):** No idle to fill → drop topic.
- **MIXED:** Small overlap window (1-2 ops per layer) recoverable → 2-5% e2e gain, marginal.

**Implementation if validated:**
- Add second `MTLCommandQueue` in `Dispatcher`.
- New `qwen_dense.rs::forward_decode_speculative()` path:
  - Per-layer event chain.
  - Speculative N+1 issue on second queue after N's residual write.
  - Fence at N+1 attention input.
- Behind `DISMANTLE_QWEN_SPECULATIVE_KERNEL=1`.

**Ceiling estimate:** 5-15% e2e if Topic 2 says gap is idle; 0% if frontend-busy.

**Risks / interactions:**
- **Hard dependency on Topic 2 findings.** Do not implement before Topic 2 audit completes.
- Second command queue may compete for command-processor frontend → could *worsen* serialization.
- Conflicts with megakernel (megakernel eliminates inter-layer boundaries → speculative co-issue becomes intra-megakernel, requires kernel-level scheduling).

**Effort:** Audit 1-2 days (after Topic 2). Implementation 4-8 days.

---

### Topic 8: Q4_K Hardware-Friendly Layout (silicon alignment)

**Claim:** Q4_K super-block layout straddles cache lines, doubling fetch overhead per sub-block. Custom 128-byte-aligned layout with embedded scales could 1.1-1.3× kernel throughput beyond predec/Q4K_FAST.

**Silicon basis:**
- M3 GPU cache line: 128 bytes.
- Q4_K super-block (per llama.cpp gguf spec): 256 elements × 4 bits = 128 bytes weights + 12 bytes scales/headers = 140 bytes total.
- 140-byte super-block straddles 2 × 128-byte cache lines → 2 fetches per super-block where 1 should suffice.
- Sub-block (32 elements × 4 bits = 16 bytes) granularity matches simdgroup 32-thread access pattern.
- L1 hit rate depends on cache-line reuse — straddled blocks cause spurious evictions.

**Current dismantle state:**
- Predec (`build_predec_2026_05_25`): extracts sub-block scales into separate sidecar buffer. Main data is 128-byte aligned. Scales are now scattered → cold cache → some win from main data alignment, some loss from scale fetches.
- Q4K_FAST (`build_q4k_fast_2026_05_25`): 160-byte sub-block-contiguous layout. 91% bit-identical (lower than 100% for predec) — quality concern per `q4k_fast_divergence_analysis_2026_05_26.md`.
- Q4K_FAST RSS cost: ~1.5 GB sidecar (significant memory cost).
- Neither layout achieves "scales colocated with data in same cache line."

**Audit steps:**
1. **L1 hit-rate profiling:** Instruments GPU Memory counter on Q4_K GEMV at decode shape. Compare baseline, predec, Q4K_FAST.
2. **Cache-line straddle count:** instrument kernel with shared atomic counter for cache-line crossings per super-block. (May require synthetic test variant.)
3. **128-byte aligned with embedded scales prototype:**
   - Layout: 112 bytes weights + 16 bytes scales/header per cache line. 256 elements / 112 bytes = 1.83 cache lines per super-block — still straddles.
   - Alternative: 128 elements × 4 bits = 64 bytes weights + 4 bytes scale + padding = 1 cache line per *half*-super-block. Doubles header overhead but eliminates straddling.
   - Offline weight repack tool extension.
4. **Bench paired vs predec (current default).**
5. **Quality gate:** corpus N=100 bit-identical sweep (per W4A8 protocol).

**Expected findings:**
- **GO (Q4K_v5 wins):** 128-byte-aligned-with-embedded-scales layout delivers >5% kernel speedup AND >95% bit-identical at N=100. → Adopt as new default after predec.
- **DEAD:** No further alignment win; predec is local optimum.
- **MIXED:** Win on kernel, loses on quality → hold pending quality fix.

**Implementation if validated:**
- New layout: `tools/repack_q4k_v5.py` — offline weight repack.
- New kernel variant: `gemv_q4k_v5_aligned.metal`.
- Wire behind `DISMANTLE_QWEN_Q4K_V5=1` (mutually exclusive with predec).
- Parity + corpus quality gates.

**Ceiling estimate:** 1.1-1.3× on Q4_K GEMV (which is 50% of decode kernel time) = +5-15% e2e. Compounds with Topic 6 Dynamic Caching.

**Risks / interactions:**
- Layout changes are weight-format changes — disk/RSS cost, sidecar files.
- Three layout options (predec, Q4K_FAST, v5) will fragment kernels — need clear default + opt-in strategy.
- Quality may regress on outlier weights (similar to Q4K_FAST 91% divergence).
- Compounds with Topic 6 — tune together.

**Effort:** Audit 2-3 days. Implementation 4-6 days incl. quality sweep.

---

## 5. Composite stack & scheduling

Topics interact. Recommended audit ordering (parallelizable where independent):

**Phase A1 — parallel audits (no dependencies):**
- Topic 1 (bandwidth ceiling) — must complete first; sets the reference.
- Topic 3 (TLB / pages) — independent.
- Topic 8 (Q4_K layout) — independent.

**Phase A2 — depends on A1:**
- Topic 2 (command-processor gap decomposition) — needs Topic 1 baseline.
- Topic 4 (cache residency) — needs Topic 1 baseline.
- Topic 6 (Dynamic Caching) — needs Topic 1 baseline.

**Phase A3 — depends on A2:**
- Topic 7 (speculative co-issue) — depends on Topic 2 gap-decomposition result.

**Phase A4 — independent but high-effort:**
- Topic 5 (cross-engine concurrency) — can start any time; longest audit budget.

**Implementation precedence (if all audits GO):**
1. **Megakernel completion** (per Topic 2) — biggest single lever, already in progress.
2. **Heap-residency v2 + super-pages** (Topics 3 + 4) — together, eliminates cache cold-start that megakernel depends on.
3. **M3 Dynamic Caching variants** (Topic 6) — kernel-level changes; lifts ceiling further.
4. **ANE-FFN concurrent** (Topic 5) — highest ceiling but highest risk; longest timeline.
5. **Q4_K v5 layout** (Topic 8) — incremental, ship after layouts stabilize.
6. **Speculative co-issue** (Topic 7) — last; depends on megakernel + Topic 2 result.

**Combined ceiling if all topics deliver mid-range:**
- Topic 2 megakernel: 26.6 → 50 dec_tps (1.88×)
- Topic 3+4 cache: ×1.1 → 55
- Topic 6 Dynamic Caching: ×1.2 → 66 (now ABOVE LPDDR wall; reality-capped at 63)
- Topic 5 ANE concurrent: lifts wall via separate bandwidth → 63 → 90+ if private DMA
- Topic 8 layout: ×1.1 → +marginal at this point (already kernel-saturated)

**Realistic stretch target: 60-90 dec_tps (2.3-3.4× current).**
**Conservative target: 45-55 dec_tps (1.7-2× current).**

---

## 6. Audit deliverables (Phase A output)

Each topic must produce, under `reports/silicon_audit/<topic_N_name>/`:

1. `audit_log.md` — chronological log of audit steps run, commands, results.
2. `measurements.json` — structured data (bandwidth_GB_s, l2_hit_rate, register_count, etc.).
3. `verdict.md` — one of:
   - `## GO\n**Justification:** ...\n**Ceiling estimate:** ...\n**Recommended next steps:** ...`
   - `## HOLD\n**What's unclear:** ...\n**To unblock:** ...`
   - `## DEAD\n**Cause of death:** ...\n**One-line summary for dead_levers.md:** ...`
4. `instruments_traces/` — `.trace` bundle if MST was captured.

Final compilation: `reports/silicon_audit/_findings.md` — top-level summary:
- Top 2-3 GO items with ceiling
- All DEAD items with one-line cause-of-death (also append to `reports/dead_levers.md`)
- Recommended implementation order
- Critical assumptions revealed

---

## 7. Halt conditions

Per `CLAUDE.md` haul rules:

- **Per-item soft ceiling:** 60 minutes during audit phase. If a topic's audit can't complete in 60 min, mark HOLD with blocker noted.
- **Haul hard ceiling:** 4 hours per session. Audit phase spans multiple sessions; one session = 2-3 topics typical.
- **Halt budget:** 2 halts per session ends the session. First halt → next topic. Second halt → write closeout, stop.
- **Memory-coexist:** if slm running, follow `tools/haul/coexist.sh` probe + watch protocol. Microbenchmarks (Topics 1, 3 partial, 5a, 6 partial) may need clean GPU window — coordinate.
- **Instruments-blocked topics (2, 3, 4, 6):** if Instruments unavailable or restricted, halt those items with `reason: instruments_unavailable`. Do not fabricate findings from code-only inference.
- **Build hygiene:** any code in audit phase is for measurement only — keep in `crates/*-spike/` or feature-gated. Do not touch production paths until Phase C.
- **Per-commit rule:** if audit produces a one-off measurement crate, that's a single commit `audit: silicon T<N> <topic name>` with the measurement code + findings doc.

---

## 8. Memory entries to read before starting

The next session must read these memory entries to ground the plan in prior findings:

- `decode_gap_anatomy_2026_05_24` — establishes the 36 ms gap.
- `gpu_us_accuracy_verified_2026_05_24` — proves gap is real, not measurement artifact.
- `icb_production_scale_2026_05_24` — kills CPU-encode-only ICB; explains why Topic 2 needs GPU-side ICB.
- `qkv_concurrent_2026_05_24` — partial QKV concurrent at +1.68%, reinforces single-queue ceiling.
- `pso_transition_dead_2026_05_24` — third host-side hypothesis ruled out.
- `amx_feasibility_2026_05_24` — AMX GEMV DEAD; informs Topic 5b re-test scope.
- `build_megakernel_day3_2026_05_25` — current megakernel state; informs Topic 2 implementation.
- `build_predec_2026_05_25` — predec layout context; informs Topic 8.
- `build_q4k_fast_2026_05_25` + `q4k_fast_divergence_analysis_2026_05_26` — Q4K_FAST context.
- `build_heap_residency_2026_05_25` — heap-residency v1 revert; informs Topic 3 v2 design.
- `composition_decision_matrix_2026_05_26` — current default-on state.
- `path_to_30_findings` — MoE v3 register-pressure data; informs Topic 6.

---

## 9. Risk register

| Topic | Top risk | Mitigation |
|---|---|---|
| 1 | Instruments bandwidth gauge unavailable | Fall back to computed model; record methodology |
| 2 | MST capture restricted in current environment | Synthetic dispatch-count microbench instead; record limitations |
| 3 | Super-page alloc fails under memory pressure | Fallback to 16 KB pages; document failure mode |
| 4 | SLC residency claims unverifiable (no direct API) | Use second-order proxies (decode wall delta at varying ctx); flag confidence |
| 5 | ANE bandwidth topology unknown | Audit step 5a is the gate; do not build CoreML harness until 5a resolves |
| 6 | M3 register/occupancy report may not predict runtime | Cross-check with actual occupancy counter |
| 7 | Depends on Topic 2 result | Do not start before Topic 2 completes |
| 8 | Quality regression on outlier weights | Mandatory N=100 corpus quality gate before any ship |

---

## 10. What this plan does NOT cover (scope-out)

- **Algorithm-level wins:** Eagle5 spec-decode (separate workstream, head not trained), lookahead/Jacobi decoding (Track C blocked), continuous batching (Track E design only). These compose with silicon wins but aren't silicon-architecture levers.
- **Pure software wins:** more aggressive kernel fusion within current dispatch model, FFI-binding micro-optimizations, Rust-level zero-copy improvements. Sub-1% ceilings.
- **Model-level changes:** further vocab pruning, mixed-precision (some layers fp16, others Q4), distillation. These are quality-gated separate workstreams.
- **Cross-host optimizations:** distributed decode, KV-cache offload to disk. Not single-node decode focus.

---

## 11. Single-session quick-start (if next session is short)

If next session has limited time, attack in this order for maximum signal:

1. **Topic 1 audit (1-2 hr)** — establishes ceiling; everything else relative.
2. **Topic 2 partial: synthetic dispatch-count microbench (1 hr)** — answers "is gap frontend-busy or idle" without Instruments.
3. **Topic 5a only: ANE bandwidth probe (2-3 hr)** — gates the entire highest-ceiling topic.
4. **Findings memo (30 min)** — what we learned + what to attack next session.

This gets the three highest-leverage audit signals in one session.

---

## 12. Sign-off contract

Next session's responsibilities:

- ✅ Read all "Memory entries to read" before starting any audit.
- ✅ Execute Phase A audits in dependency order; never skip a dependency.
- ✅ Produce per-topic deliverables in `reports/silicon_audit/`.
- ✅ Compile `_findings.md` before any implementation.
- ✅ Present findings to user via terminal output AND committed `_findings.md`.
- ✅ Wait for user GO before Phase C implementation.
- ✅ Append every DEAD verdict to `reports/dead_levers.md` with one-line cause-of-death.
- ✅ Follow `CLAUDE.md` haul rules: inline git identity, single-purpose commits, halt budget, evidence triples for any gate.
- ❌ Do NOT modify this plan during execution. If audit reveals plan errors, write `reports/silicon_audit/_plan_amendments.md` for attended review.
- ❌ Do NOT skip the audit phase. The whole point is to not implement based on architectural hypotheses without verification.
- ❌ Do NOT pursue scope-outs in Section 10.
- ❌ Do NOT ship anything without parity gate (atol=1e-3 fp16 + bit-identical 3-tok) and bench paired n≥5 with `--strict`.

End of plan.
