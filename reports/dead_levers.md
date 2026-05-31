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

## 🪦 Cross-layer weight delta-encoding (Bible §8.1 L1.3)

**Status:** killed 2026-05-30 by offline weight analysis (before any kernel)
**Evidence:** `reports/oracle_interlayer_delta.md` + `tools/bench/oracle_interlayer_delta.py`. Across all 7 tensor types at layer pairs 0→1 / 17→18 / 34→35 of Qwen2.5-3B: cosine(W[L],W[L+1]) ≈ 0 (mean +0.0003, |max| 0.007 — layers essentially orthogonal); delta std / orig std = **1.61** (up to 2.9 on FFN) so the delta is ANTI-compressible (quantizing D at equal error costs MORE bits, not fewer); delta top-64 SVD energy = 0.23 (full-rank); optimal affine W[L+1]≈α·W[L]+D gives **α*≈0** (a learned gain buys nothing). 0/7 tensor types beat native Q4_K bits. Textbook well-trained transformer (each layer a distinct transform). Peak RSS 1.59 GB.
**Resurrection check:** only on a model with deliberately tied/correlated layers, or one TRAINED for cross-layer structure (Bible §8.1 L5.1 "heal the model into the engine"). Post-hoc extraction on stock Qwen is dead.

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
**Update 2026-05-30 (L2.2 oracle, `reports/oracle_coactivation_permute.md` + `tools/bench/oracle_coactivation_permute.py`):** paths (1) and (2) are now TESTED and DEAD. Per-neuron activations were reconstructed from the GGUF gate/up weights (bit-exact, rel-L2 1.7e-7 vs the captured block reductions): 99% FFN-output energy needs **39–53% of all 11008 neurons/token with 0 permanently cold** (no static hot/cold split → path 1 dead), and best-case offline **co-activation permutation** lifts skippable blocks @99% recall to only **1.5%/0.8%/0.3%** at block {32/64/128} (vs ~30% bar) because the top-200 hot neurons reshuffle every token (~22–25% Jaccard) so no STATIC permutation can pack them (→ path 2 dead). Only resurrection path left is **trained-for** sparsity (Bible §8.1 L5.1), not post-hoc extraction.

---

## 🪦 Host-side per-dispatch overhead (concurrent encoder, PSO batching, gap-closing)

**Status:** family exhausted 2026-05-24 — the decode "gap" is real GPU-side, not host
**Evidence:** Three host-side hypotheses for the ~36 ms/token inter-dispatch "gap" all ruled out. (1) Q/K/V concurrent encoder (`begin/end_concurrent_group`, `DISMANTLE_QWEN_CONCURRENT_QKV=1`): bit-identical but paired only **+1.68%**, below the +5% ship gate. (2) PSO-transition batching: 200 dispatches with 199 PSO transitions runs **1.06×** vs identical-kernel — essentially free. (3) CPU encode is **0.51%** of wall ([[v230-icb-dead]], [[cpu-gpu-pipelining-audit]]). And `gpu_us` is **accurate** at production workload (host_wall/Σgpu = 1.03×), so the gap is NOT a measurement artifact — it is real GPU-side time. **Conclusion: every host-side per-dispatch lever (ICB, concurrent encoder, PSO batching, megakernel-for-dispatch-count) caps below the ship gate.** The real lever is kernel bandwidth efficiency (Bible Stage 2, ~41%→60%+ of peak), not dispatch overhead.
**Killing memory:** [[gpu-us-accuracy-verified-2026-05-24]], [[pso-transitions-dead-2026-05-24]], [[qkv-concurrent-2026-05-24]], [[decode-gap-anatomy-2026-05-24]]
**Resurrection check:** only if a future kernel restructuring pushes dispatch count per token far higher (CPU encode crossing ~1 ms/token), or Apple ships a materially cheaper concurrent-encode primitive. The gap itself is closed by making each kernel faster (Stage 2 simdgroup-matrix decode, worklist 1.7), not by removing dispatch overhead.

---

## 🪦 ICB (Indirect Command Buffer)

**Status:** killed 2026-05-14 by 0.51% CPU-encode budget
**Evidence:** `DISMANTLE_TCB_TRACE=cpu` measurement: per-token steady-state CPU encoding = 0.22 ms = 0.51% of wall. Ideal ICB ceiling +0.51% e2e; realistic +0.23%. Below the +5% ship gate.
**Killing memory:** [[v230-icb-dead]]
**Resurrection check:** if dispatch count per token grows substantially (e.g. via per-expert serial dispatch or new fused-kernel restructuring), re-measure CPU encode budget. If it crosses 1 ms / token, the lever becomes plausible again.
**Pre-flight gate:** always run `DISMANTLE_TCB_TRACE=cpu` and check p50 weighted per-kernel encode sum vs Off-mode wall before proposing an ICB / megakernel / pipeline-replay lever.

---

## 🪦 Low-rank + compressible residual codec (Bible §8.1 L1.4)

**Status:** killed 2026-05-30 by offline byte-budget oracle (before any kernel)
**Evidence:** `reports/oracle_lowrank_codebook.md` + `tools/bench/oracle_lowrank_codebook.py`. Qwen2.5-3B weights are not low-rank: top-64 SVD captures only **3–9%** (FFN) to **~26%** (attn) of Frobenius energy; residual std stays **~90–99%** of the original (median 0.95). Since SVD removes no structure, residual@2–3b ≈ raw-quant@2–3b, so the f16 U,V are **pure dead overhead → strictly worse than plain low-bit quant**. (The raw byte ratio looked like a win only because it stored the residual at <4.5b — illusory.) Build at most one byte-cut codec; this isn't it.
**Resurrection check:** only on a model trained to be low-rank (L5.1). The surviving byte-cut codec is **QTIP** (lookup-free bitshift trellis) — a real byte cut AND gather-free; advance it to the GPU/quality lane.

---

## 🪦 Learned per-model codebook (Bible §8.1 L1.5)

**Status:** killed 2026-05-30 at the Apple-GPU feasibility gate (before quality eval)
**Evidence:** `reports/oracle_lowrank_codebook.md`. A raw k-means codebook is an index→value table = per-element **RANDOM LUT gather** — exactly the IQ-quant pattern Apple GPUs punish (no hardware gather), even with a 16/256-entry threadgroup-resident table; k=256 is 8-bit (no compression). A 1-D learned grid's MSE is worse per-bit than fixed grids (1.87× Q4_0, 32.8× Q8_0) because it lacks per-block scales. The binding constraint is **decode feasibility, not quality** — killed before any KL/PPL eval, per the Bible's "kill it at the feasibility gate" rule.
**Resurrection check:** only a LOOKUP-FREE learned code (QTIP's bitshift trellis — codes are computed, not gathered). A gather-based codebook stays dead on Apple Silicon.

---

## 🪦 Mixed-precision / W4A8 as a default decode path

**Status:** held (not shipped) 2026-05-24 — quality-blocked + below ship gate
**Evidence:** Per-block int8 activation × Q4_K GEMV is correct and fast in microbench (−34% kernel time, [[w4a8-prototype-2026-05-24]]) but at the model level it fails on two axes. Quality: N=100 corpus = **20% bit-identical** at 32-tok greedy ([[w4a8-corpus-quality-2026-05-24]]). Perf: paired decode **1.115×** — below the 1.20× ship rule, and the fused-quantize variant landed identical 1.116× ([[w4a8-production-held]], [[w4a8-fused-quantize-held]]). Composition: every W4A8 combo is **sub-additive** vs predec-alone (predec+w4a8 = 1.151× < predec 1.340×, [[composition-decision-matrix-2026-05-26]]). Naive per-tensor mixed precision likewise lost in the M5 stack matrix.
**Killing memory:** [[composition-decision-matrix-2026-05-26]], [[w4a8-corpus-quality-2026-05-24]], [[w4a8-production-held]]
**Resurrection check:** needs BOTH (1) a logit-streaming quality metric — bit-identical is too strict; cosine/KL on logits may show acceptable quality — AND (2) a clean low-bit source: the byte-cut path requires AWQ-from-f16, not requant-from-Q4_K ([[bible-execution-2026-05-30]]). Held infra stays behind `DISMANTLE_QWEN_W4A8=1`.

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

## 🪦 Predec 4-row ILP (`_4r`) as a default — speculative, unvalidated

**Status:** parked 2026-05-30 — a guess not grounded in valid profiling
**Evidence:** `gemm_q4_k_v4_predec_4r` (4 accumulator chains) was added on the theory that the decode GEMV underfills the GPU at 2 rows/simdgroup. But the homemade TCB trace that motivated it is split-CB-distorted and the §1 methodology gate rejects it ([[bible-execution-2026-05-30]]), so the "occupancy-starved" premise is unproven. The validated win is `_2r` (+6.2% bit-identical, default-on); `_4r` is bit-identical but **not adopted** and stays opt-in (`DISMANTLE_QWEN_PREDEC_4R=1`).
**Killing memory:** [[bible-execution-2026-05-30]], [[path-to-50-gap-corrected-2026-05-29]]
**Resurrection check:** re-evaluate ONLY after worklist 0.1 (xctrace profiling export) gives valid per-kernel occupancy/stall data. Adopt `_4r` only if profiling shows the predec GEMV is occupancy-limited (not bandwidth-limited) at 2 rows — the Bible says decode is bandwidth-bound (~41% of peak), which predicts more ILP will NOT help. The canonical example of "don't guess kernel geometry without valid profiling."

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
- [[gpu-us-accuracy-verified-2026-05-24]] — proved the ~36 ms/token decode gap is real GPU-side, killing the host-side per-dispatch overhead family
- [[composition-decision-matrix-2026-05-26]] — predec wins quality+perf; every W4A8 combo is sub-additive
- [[bible-execution-2026-05-30]] — §1 gate enforced; homemade trace rejected (motivates 0.1 profiling); 4r unvalidated
