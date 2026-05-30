# Dead levers — do not re-spawn

A consolidated index of levers that have been investigated and ruled out.
Each entry has the killing evidence; before re-spawning, verify the
evidence is still current (sometimes the killing assumption was a
profiling artifact that resolved).

Ordered alphabetically by lever name.

---

## 🪦 CPU+GPU pipelining (sampler/tokenizer overlap with forward)

**Status:** killed 2026-05-22 by code audit (Session I)
**Evidence:** Greedy hot path already encodes all 27 layers + final norm + LM head + GPU-side argmax into one TokenCommandBuffer (`deepseek_v2.rs:2531-2758`), one `commit_and_wait`, then reads 4 bytes (token id) from shared memory. Sampler is GPU-side via `sample_argmax_f32_tcb` — no CPU sampler in default greedy bench. KV writes are GPU-side via `kv_append_f32`; no CPU sync. Post-commit CPU work is `kv.seq_len += 1` + sink callback (µs-scale). Only real overlap gap is pre-commit CPU encode = 0.22 ms = 0.51% of wall (inherited from [[v230-icb-dead]]). Ceiling ≈ +0.14 dec_tps; well below any ship gate.
**Killing memory:** [[cpu-gpu-pipelining-audit]]
**Resurrection check:** if a sampling mode other than greedy/argmax becomes the primary (top-k/temperature/Mirostat with CPU-heavy logic), re-measure. Also re-measure if dispatch count grows enough that CPU encode crosses 1 ms/token (same gate as ICB).

---

## 🪦 Eagle5 v1 (routing-mask predictor)

**Status:** killed 2026-05-21 by corpus analysis
**Evidence:** `artifacts/calibration/analysis/expert_load_per_layer.json` — per-layer balance scores 0.987–0.995 across all 26 MoE layers. Hottest expert at any layer is 2-4% load vs uniform 1.56%. No concentration to exploit.
**Killing memory:** [[corpus-complete-analysis-landed]]
**Resurrection check:** if the calibration is re-run on a different corpus / chat template and the balance scores drop below ~0.95, this becomes interesting again. The current 0.987-0.995 is essentially uniform — a perfect mask predictor would be useless.
**Pivoted to:** Eagle5 v2 (activation-sparsity predictor) — see `reports/eagle5_v2_wiring_handoff.md`.

---

## 🪦 f16 residual stream (Phase Z-1)

**Status:** killed 2026-05-11 by accumulated error after 27 layers
**Evidence:** `wedge_f_active` path is correct per-kernel (5 tests pass at 1e-3 fp16 atol), but full model produces garbage tokens. Root cause: residual |x| ≈ 5-10; f16 epsilon ≈ 1e-3; error after 27 layers ≈ 0.27/element → corrupts logits.
**Killing memory:** [[v110-path30-findings]]
**Resurrection check:** bf16 might work (8-bit exponent matches f32 range). Only worth retrying if a per-kernel benchmark first proves the bandwidth saving is large enough to justify the rewrite cost.

---

## 🪦 FFN contextual sparsity at block-256 (Track B "the breakthrough")

**Status:** killed 2026-05-30 by Step-2 oracle measurement (before kernels)
**Evidence:** `reports/ffn_sparsity_track_b_gate.md` + `reports/ffn_sparsity_gate.json`. 800 real decode tokens × 36 layers (`_capture/q3b_ffn.bin`). Oracle (best-case) skippable 256-blocks at 99% FFN-output recall = **0.2%** (0.1% bytes/token); even at quality-destroying 90% recall = 4.2%. Handoff projected ~65%. Root cause is NOT lack of sparsity: participation ratio (L2/max)² = **5.6 active channels/256-block (~2.2%)** — q3b IS neuron-sparse like ReLU Deja-Vu models — but the ~5 active neurons/block are **scattered**, so every 256-block has a few and none is droppable. Granularity mismatch: sparsity is neuron-fine, byte-skipping needs block-contiguous.
**Killing memory:** [[ffn_block_sparsity_dead_2026_05_30]]
**Resurrection check:** Only at FINER granularity, and only as attended R&D: (1) neuron-granularity re-capture → test PowerInfer static hot/cold split (current capture stored per-block reductions only, can't measure per-neuron frequency); (2) offline co-activation permutation to cluster co-firing neurons into contiguous blocks (risk: co-activation is input-dependent → low yield); (3) sparse Q4_K layout for neuron gather/scatter (Q4_K 256-super-block makes single-column gather non-aligned/random-access — historically not BW-favorable on Apple Silicon). Block-256 predictor itself stays dead. Capture tooling (`--capture-ffn`, `pack_ffn.py`, `measure_ffn_sparsity.py`) is kept for any of these.

---

## 🪦 ICB (Indirect Command Buffer)

**Status:** killed 2026-05-14 by 0.51% CPU-encode budget
**Evidence:** `DISMANTLE_TCB_TRACE=cpu` measurement: per-token steady-state CPU encoding = 0.22 ms = 0.51% of wall. Ideal ICB ceiling +0.51% e2e; realistic +0.23%. Below the +5% ship gate.
**Killing memory:** [[v230-icb-dead]]
**Resurrection check:** if dispatch count per token grows substantially (e.g. via per-expert serial dispatch or new fused-kernel restructuring), re-measure CPU encode budget. If it crosses 1 ms / token, the lever becomes plausible again.
**Pre-flight gate:** always run `DISMANTLE_TCB_TRACE=cpu` and check p50 weighted per-kernel encode sum vs Off-mode wall before proposing an ICB / megakernel / pipeline-replay lever.

---

## 🪦 MLA Phase 4 simdgroup attention rewrite

**Status:** killed 2026-05-22 by clean paired bench (Session D)
**Evidence:** Cherry-picked the shader rewrite (recovered from dangling stash `c863bba`, original branch `claude/mla-phase4-experiment` was pruned). Parity GREEN (`integration_greedy_64`, `v1_1_phase4D_spec_exact_mode` both PASS). `quick_bench.sh` paired A/B/A: OLD 24.90, NEW r1 24.27, r2 24.46 → **-1.7% to -2.5% reproducible regression**. The simd_sum-per-vi pattern adds enough overhead to swamp the "threads 128..255 idle" structural win. Per `per_kernel_time_breakdown.md`, MLA attention is only 2.4% of decode time anyway, so even a perfect rewrite caps at +2.4% — not worth pursuing.
**Killing memory:** [[mla-phase4-resurrected]] (supersedes prior [[mla-phase4-queued]])
**Resurrection check:** would need a new simdgroup intrinsic on Apple Silicon that materially lowers the simd_sum overhead, AND an attention-heavy workload (long context) where the 2.4% share grows. Neither is on the horizon for V2-Lite decode.

---

## 🪦 MoE megakernel (gate+up+SiLU+down fused)

**Status:** killed 2026-05-14
**Evidence:** gate+up+SiLU already fused in `moe_batched_gemm_q4_indexed_v2t_gu_v2` (shader line 626). Only remaining DRAM hand-off (y_act between gate_up_silu and down) is 1.12 MiB/token = 0.04% of wall. Cross-TG synchronization makes deeper fusion infeasible.
**Killing memory:** [[v230-icb-dead]]
**Resurrection check:** would need a new Apple-Silicon sync primitive that allows cross-TG barriers; not in current Metal.

---

## 🪦 MoE serial route dispatch (one expert at a time)

**Status:** killed 2026-05-11
**Evidence:** Hypothesis was L2 thrashing from 6 simultaneous scattered expert streams. Measured: single-TCB serial dispatch = 50 ms/token (WORSE than 44 ms parallel). Per-encoder Metal overhead (~0.01 ms × extra dispatches) cancels the L2 benefit. 200 extra encoder setups across 27 layers cost ~2 ms.
**Killing memory:** [[v110-path30-findings]]
**Resurrection check:** if Metal introduces zero-cost encoder switching, re-measure. Code is still in tree as `v2t_gu_serial` option.

---

## 🪦 Phase Y — sumy-trick Q4_K v3 (256 threads/TG, register-pressure-heavy)

**Status:** killed 2026-05-11 by -14% regression
**Evidence:** `moe_batched_gemm_q4_indexed_v3` (64 threads/TG, 4 rows/simdgroup, sumy trick) is parity-correct but 14% slower than v2. `ds[4][8]` + `dm[4][8]` = 64 floats/thread of local scale arrays → register pressure → occupancy collapse.
**Killing memory:** [[v110-path30-findings]]
**Resurrection check:** the "keep v2 geometry (256 threads/TG, 1 row/simdgroup) but add sumy trick with only `ds[8]`+`dm[8]`+`xl[8]`+`sumy[8]` = 32 floats overhead" idea was never tried; if Q4_K bandwidth becomes the dominant bottleneck again, that variant might escape the register-pressure trap.

---

## 🪦 Q5_0 simd_shuffle byte broadcast (audit fix #1)

**Status:** killed 2026-05-14 by -3.5% regression
**Evidence:** Bit-identical 3-token parity but benches -3.5% trimmed_mean. Apple's HW already coalesces redundant lane-pair byte loads; simd_shuffle overhead exceeds the savings.
**Killing memory:** [[feedback-kernel-parity-gate]]
**Resurrection check:** if Apple silicon coalescing changes (new GPU gen), retry; otherwise the HW already does this.

---

## 🪦 Q8-KV layer-differential precision

**Status:** killed 2026-05-21 by uniform routing
**Evidence:** Per-layer routing balance 0.987-0.995 means there's no signal driving "which layers' KV needs higher precision". The intuition was that layers with concentrated routing might tolerate lower-precision KV; calibration shows no concentration exists.
**Killing memory:** [[corpus-complete-analysis-landed]]
**Resurrection check:** if a new model shows skewed routing (balance < 0.95), retry. Current uniform-Q8 stays per [[q8-kv-landed]].

---

## 🪦 Speculative-decode ExactShared as-is

**Status:** regression confirmed 2026-05-11, structurally infeasible without batched verify or 10-20 ms draft
**Evidence:** dec_tps = 0.11 vs 18 baseline. 7.3% draft acceptance × 235 ms per spec step (5 verify passes × 47 ms) = 4.5 tps ceiling at any acceptance rate. Sequential verify cannot win.
**Killing memory:** [[v110-path30-findings]], [[path-to-100-repath]]
**Resurrection check:** parallel/batched verify OR a 10-20 ms draft model OR fixing the GPU-clock-down between small-CB draft phases. The eagle5 v2 head + a re-designed verify is the resurrection path — see `reports/eagle5_v2_wiring_handoff.md` and [[path-to-100-repath]] Track 2.

---

## 🪦 LM head simdmat as a tps lever (Phase X v1.1.0)

**Status:** killed 2026-05-11 by mis-estimated cost share
**Evidence:** LM head was ~4% of decode time, not 70% as the spec assumed (source of 70% estimate unknown — possibly contaminated trace). Implementation landed (`gemv_f16_simdmat`) and is correct, but the target isn't the bottleneck.
**Killing memory:** [[v110-path30-findings]]
**Resurrection check:** the kernel is in tree; useful if LM-head cost share ever rises (e.g. after a much larger vocab change). Today it's parked.
**Lesson:** never invest in a lever before measuring its cost share. Run per-kernel time breakdown first (per `reports/per_kernel_time_2026-05-20.md`).

---

## Pre-spawn checklist for any new lever

Before opening a wedge, audit:

1. **Is this lever in this document?** If yes, read the resurrection check.
2. **Have you measured the cost share?** Per-kernel time breakdown
   (`reports/per_kernel_time_2026-05-20.md` is the latest snapshot)
   or fresh `DISMANTLE_TCB_TRACE=cpu` run. If the lever can save at
   most N%, and N < your ship gate, it's dead before you start.
3. **Does it gate on a calibration insight?** Run
   `tools/training/analyze_corpus.py` first if so. Routing-balance
   killed two levers in one analysis run.
4. **Does it depend on a downstream system that's already
   regressing?** (Spec-decode runtime, batched verify, etc.) Fix the
   downstream regression first or accept the lever ships dormant.

---

## Cross-references

- [[corpus-complete-analysis-landed]] — calibration that killed Q8-KV-layer-diff + eagle5-routing
- [[v110-path30-findings]] — kill notes for Phase X/Y/Z + serial dispatch + spec-decode
- [[v230-icb-dead]] — ICB + MoE megakernel kill notes
- [[feedback-kernel-parity-gate]] — Q5_0 simd_shuffle kill + the suffix-matcher bug story
- [[path-to-100-repath]] — current spec-decode runtime regression context
