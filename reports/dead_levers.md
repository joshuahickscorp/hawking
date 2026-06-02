# Dead levers — the canonical kill-ledger — do not re-spawn

**This is the one canonical kill-ledger for dismantle.** It merges (a) the
throughput-bible / dead-lever kills, (b) the 16-solution silicon-architecture
audit (transcribed from the now-deleted untracked `silicon-builds/` tree), and
(c) the 2026-06-01 Colab sub-Q4 byte-cut quality gates. `plans/bible_archive.md`
§8.3.1 (the Kill Protocol) anoints this file as the full authoritative registry;
the bible carries only the four worked-example Phase-A kills. Update this file
whenever a lever dies.

Each entry has the killing evidence; before re-spawning, verify the evidence is
still current (sometimes the killing assumption was a profiling artifact that
resolved) and read its resurrection check. **Never re-test a recorded Type-1
kill** (its death is a fact about reality); **never resurrect a Type-2 on vibes**
— only behind its named, cheap oracle.

The bible/dead-lever entries below are ordered alphabetically by lever name; the
silicon-audit and Colab-verdict sections follow.

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
**Update 2026-05-31:** the data-aware (activation-weighted) cross-layer reframe was TESTED (`reports/oracle_dataaware_lowrank.md`): data-weighted cross-layer cosine ≈ 0 → NO-GO too. Both weight-space and data-space cross-layer reference are dead. (Same oracle settled L1.4 — see below.)

---

## 🪦 EAGLE-3 trained draft head (Eagle5 v3, axis-3 speculation)

**Status:** NO-GO concluded 2026-05-31 — doubly confirmed (offline held-out + on-device)
**Evidence:** Retrained num_blocks=2 head on a bigger/diverse 40-shard Q4_K_M capture, evaluated HELD-OUT (`~/Downloads/eagle3_train_result.json`): **τ=0.877** (gate 2.5), per-pos accept [0.523, 0.195, 0.097, 0.062]. The corpus fix DID improve generalization (held-out depth-1 33%→52%) but nowhere near useful. On-device paired bench (`eagle5_paired_bench.sh`, new head, Qwen-3B + locked env, code prompt): baseline **36.9 dec_tps**; spec K=2/4/8 = **14.9 / 11.1 / 7.6** (0.40×/0.30×/0.21× — net-negative, worse with larger K). On-device depth-1 accept **6.5%** vs PyTorch held-out 52% (~8× gap ⇒ a residual head↔runtime forward mismatch remains — but it's MOOT: even at the 52% offline ceiling, τ=0.88 is sub-gate). The **free n-gram draft (τ=1.43) beats the trained head (0.877)** on code.
**Killing memories:** [[phase-a-oracles-2026-05-30]]; handoff `plans/eagle_forward_parity_handoff.md`.
**Resurrection check:** do NOT re-train the EAGLE head expecting a win without an oracle first showing achievable τ≥2.5 on the target workload. The trained-head path is net-negative on Qwen-3B + code; n-gram lookahead (also sub-gate, τ=1.43) is the only spec option worth keeping warm. The forward-parity gap (6.5% device vs 52% offline) is real but not worth chasing while the offline ceiling itself fails the gate.

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

## 🪦 Phase 2.2 trivial-op dispatch fusion (rmsnorm/add/rope/memcpy merges) — dead-for-tps

**Status:** dead-for-tps 2026-06-02 — **Type-2** (the lone bit-identical fusion is real but sub-noise; the non-noise fusions are Type-1 dead or break bit-identity). Closes Phase 2.2 as a tps lever.
**Charge:** the 0.2 trace shows ~324 trivial-op dispatches/tok (rmsnorm 72 already-fused + add 108 + rope 72 + memcpy 72) = ~53% of the 616 dispatch COUNT but only ~9% of GPU time. Find the highest-leverage BIT-IDENTICAL fusion. (Wave-2 `dispatch-fusion` evaluated all three.)
**Verdict:**
- **(a) rope-q + rope-k single dispatch:** removes ~36 encode-latency dispatches/tok but ZERO DRAM round-trips (both ropes already in-place); needs a new two-buffer kernel that is not trivially bit-identical → the already-dead **host-per-dispatch-overhead Type-1 family** (CPU encode 0.51% of wall; the gap closes via faster GEMV, not fewer dispatches). DEAD.
- **(c) add+rope / add+memcpy epilogue fold:** the fused-arithmetic store (`shmem[0]+bias`) is the A10/A5-b FMA-recontraction 1-ULP trap; fails the bit-identical golden gate by construction. DEAD-by-gate.
- **(b) KV-append memcpy elision (`DISMANTLE_QWEN_KV_DIRECT`):** the ONLY bit-identical real-round-trip kill — the fused k+v `_pair` GEMV writes directly into the KV-cache seq slot (output bound at byte offset = pure pointer arithmetic), bias-add + rope-k in-place via the existing `_off` wrappers, eliding the two `memcpy_f32_off_tcb`. Proven bit-identical. But the two KV-append memcpys are **0.83% GPU-time ≈ 0.6% wall** → inside the ±3% paired-bench inconclusive floor (0.1 calibration).
**Why dead:** decode is 86.7% GEMV-bound (the two predec GEMVs at ~52% of peak); the trivial-op surface is ~9% GPU and most of THAT is irreducible in-register compute (rope rotation, bias add, norm all still execute) — only the DRAM round-trips + encode latency are removable, and they are too small. Dispatch-reduction routes to **GEMV bandwidth efficiency** — itself the recorded A5/A6/A10 **Type-1** wall — and bytes route to QTIP (also Type-1 dead). Net: short-ctx dense tps is structurally tapped; gains route to spec/stateful/long-ctx, not fusion.
**Draft (not landed):** the bit-identical KV_DIRECT lever was BUILT (draft in `reports/wave2_result.json`, stream `dispatch-fusion`) but **NOT landed** — below the ship gate, hot-path churn unjustified for ≤0.83%, and the diff had hunk-span defects. Ready to flip in at zero quality cost if a future profile ever prices the memcpy above the gate.
**Resurrection check:** revisit ONLY if a future kernel restructuring pushes dispatch count per token far higher (CPU encode crossing ~1 ms/token at ≫616 dispatches/tok — it does not today) OR KV-cache geometry grows the per-slot relocation by orders of magnitude.
**Cross-ref:** the **Host-side per-dispatch overhead** Type-1 family (ICB/concurrent-encoder/PSO/megakernel-for-COUNT); KV_DIRECT is the round-trip-elimination sub-form it does not cover, in the same sub-gate territory for the same GEMV-bound root reason.

---

## 🪦 ICB (Indirect Command Buffer)

**Status:** killed 2026-05-14 by 0.51% CPU-encode budget
**Evidence:** `DISMANTLE_TCB_TRACE=cpu` measurement: per-token steady-state CPU encoding = 0.22 ms = 0.51% of wall. Ideal ICB ceiling +0.51% e2e; realistic +0.23%. Below the +5% ship gate.
**Killing memory:** [[v230-icb-dead]]
**Resurrection check:** if dispatch count per token grows substantially (e.g. via per-expert serial dispatch or new fused-kernel restructuring), re-measure CPU encode budget. If it crosses 1 ms / token, the lever becomes plausible again.
**Pre-flight gate:** always run `DISMANTLE_TCB_TRACE=cpu` and check p50 weighted per-kernel encode sum vs Off-mode wall before proposing an ICB / megakernel / pipeline-replay lever.

---

## 🪦 KV working-set eviction (StreamingLLM/H2O/SnapKV) (Bible §8.1 L1.1)

**Status:** NO-GO 2026-05-31 by the attention-mass concentration oracle (before any eviction wiring). **Type-1** kill on Qwen2.5-3B + code context.
**Evidence:** `reports/b2_kv_working_set_oracle.md` + `tools/bench/oracle_attn_mass.py` + capture `reports/bench/attn_capture.json`. Built a default-off attention-capture instrument (`crate::stateful::attn_capture`, fed from `forward_token`'s CPU reference attention; recompute is bit-identical to `mha_decode_step`, unit-tested). Ran a real ~586-token code prompt, 917 query samples per layer, all 36 layers. **Attention mass is broadly spread, not concentrated:** holding ≥99% mass needs **78–92% of the cached positions** (median 0.80, worst layer 0.92) — a "bounded working set" would be ~539 of 586 positions, i.e. nearly the whole cache. The StreamingLLM **sinks+recent** structure does NOT hold on this model: sinks(span 4)+recent(span 128) covers only **18–73%** of mass per layer (worst layer 0.18). Per-layer behavior is heterogeneous (some layers sink-heavy e.g. L5 sink-mass 0.44; some recent-heavy e.g. L1 recent-128 0.79) so no single bounded policy clears every layer simultaneously — and the budget must hold on the hardest layer. Concentration verdict thresholds (GO: worst-layer frac99 < 0.25 AND sinks+recent ≥ 0.97) both fail by a wide margin.
**Reframe considered:** H2O (cumulative-mass heavy-hitters instead of positional sinks+recent). Also dies: if 99% mass needs ~80–92% of positions, the "heavy hitter" set is itself ~80–92% of the cache regardless of *how* you pick it — the diffuseness is a property of the distribution, not the selection rule. SnapKV (pooled-window importance) has the same problem.
**Resurrection check (named oracle):** re-run `tools/bench/oracle_attn_mass.py` on **genuinely longer** captures (16K–32K ctx) — sink/recent structure is documented to sharpen with length, and 586 tokens may understate the long-context case. Also re-test on **non-code** workloads (prose, multi-turn chat) and a **larger Qwen** (7B/14B) where the literature found StreamingLLM holds. Build the eviction bodies only if a longer/other-domain capture flips worst-layer frac99 below ~0.25. The `LosslessPolicy` escape hatch ships regardless (no-op, needs no oracle). Do not re-test the 586-token-code regime — its death is a fact about that distribution.
**Lesson:** the StreamingLLM/H2O "attention is sparse" finding is model+context-specific; it was NOT assumed for Qwen2.5-3B and the measurement says it does not transfer at short-to-mid code context. Same discipline as block-256 FFN sparsity.

---

## 🪦 Low-rank + compressible residual codec (Bible §8.1 L1.4)

**Status:** NO-GO — killed 2026-05-30 (data-free SVD, before any kernel) + **data-aware reframe RAN and confirmed dead 2026-05-31** (→ now **Type-1**; see Update below)
**Evidence:** `reports/oracle_lowrank_codebook.md` + `tools/bench/oracle_lowrank_codebook.py`. Qwen2.5-3B weights are not low-rank: top-64 SVD captures only **3–9%** (FFN) to **~26%** (attn) of Frobenius energy; residual std stays **~90–99%** of the original (median 0.95). Since SVD removes no structure, residual@2–3b ≈ raw-quant@2–3b, so the f16 U,V are **pure dead overhead → strictly worse than plain low-bit quant**. (The raw byte ratio looked like a win only because it stored the residual at <4.5b — illusory.) Build at most one byte-cut codec; this isn't it.
**Resurrection check:** only on a model trained to be low-rank (L5.1). The surviving byte-cut codec is **QTIP** (lookup-free bitshift trellis) — a real byte cut AND gather-free; advance it to the GPU/quality lane.
**Update 2026-05-31 — data-aware reframe RAN and DIED → now Type-1:** the activation-aware SVD reframe (the named Type-2 escape; ASVD/SVD-LLM on `W·C^{1/2}`) was tested (`reports/oracle_dataaware_lowrank.md` + `tools/bench/oracle_dataaware_lowrank.py`, 36 layers × {ffn_gate, ffn_up}, 800 tok/layer, 70/30 held-out). NO-GO: data-norm E64≈0.990 is an **in-sample artifact** — captured activations are effectively rank-≤64 (participation 3.1/2048, rank99% 63), so `data-E64≈1.0` holds for any weight and saves no WEIGHT bytes; held-out error blows up (0.139 vs 0.079 in-sample); 1/72 FFN tensors beat Q4_K bytes at the 0.02 gate. Lower-bound caveat (target W = dequantized Q4_K) noted, but the NO-GO is decisive. Both data-free (2026-05-30) and data-aware forms are dead → QTIP is the surviving byte-cut codec. (Re-confirmed in `reports/kill_ledger_reconciliation.md`.)

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

## 🪦 Q3_K sub-Q4 decode byte-cut (f16-predec-Q3 / cheaper-decode-Q3)

**Status:** NO-GO 2026-05-31, **clean-room confirmed** — Type-1 (reality-dead)
**Evidence:** clean `q3k_bytecut_bench` (Claude quit, `tools/bench/clean_room_batch.sh` §A): f32-predec-Q3 best-shape **33.3 GB/s = 22% of 150 GB/s peak** (vs the ~50% GO bar). Decisive tell: Q3_K is **slower in absolute µs** than Q4_predec on all 3 shapes (−37 to −43%) despite ~half the bytes — so the Q3_K GEMV is **compute/residual-bound on the inline 6-bit scale + hmask decode, NOT bandwidth-bound.** Fewer bytes buy no speed when the kernel isn't on the bus. Confirms c1f5275's 2-row-ILP finding (7–21 GB/s) at a clean absolute number. **Re-confirmed clean 2026-06-01** (36.3 GB/s best-shape = 24.2% peak; Q3 38–46% slower in µs than Q4_predec on all 3 shapes).
**Type:** Type-1 — a measured property of the Q3_K format (hmask/index residual + per-element scale decode). The f16-predec-Q3 128-B repack (`plans/cheaper_decode_q3_design_2026_05_31.md`) shaves a few more bytes but cannot flip a compute-bound kernel to BW-bound. The footprint cut (~−27/−38% bytes) is real but RAM-only, not tps.
**Type-2 reframe (the live one):** **QTIP** gather-free trellis (`plans/qtip_bytecut_design_2026_05_31.md`) — a DIFFERENT mechanism (arithmetic-coded, no hmask residual, no LUT), alive behind its own quality + decode-cost oracles. Now the **single** sub-Q4 byte-cut bet.
**Killing memory:** [[moat-status-forward-path-2026-05-31]]
**Resurrection check:** do NOT wire f16-predec-Q3 for tps. The byte-cut axis routes through QTIP only; Q3_K stays footprint-only if RAM ever becomes the binding constraint.

---

## 🪦 QTIP-on-Metal trellis decode (Phase 4.1 — decode cost gate)

**Status:** NO-GO 2026-06-02 — **Type-1 by proxy**, recorded. Closes the sub-Q4 byte-cut axis for decode speed.
**Evidence (by proxy — accepted per `scout_phase_4_1_qtip_spike.md` §4.1-0):** clean `q3k_bytecut_bench` (`clean_room_batch.sh` §A, Claude quit, re-confirmed clean 2026-06-01): f32-predec-Q3 best-shape **36.3 GB/s = 24.2% of 150 GB/s peak** (the paradigm log also records 33.4 GB/s = 22.3% from a slightly different sequence); Q3 kernels run **38–46% slower in absolute µs than Q4_K predec on all 3 shapes despite ~half the bytes** — compute/residual-bound on the inline 6-bit scale + hmask decode, NOT bandwidth-bound (see the **Q3_K sub-Q4 decode byte-cut** Type-1 entry above). Proxy chain: (a) a QTIP trellis decode **re-adds** the per-element decode ALU that predec removed — *more* per-element ALU than f16-predec-Q3, itself the leaner reframe of the dead Q3 kernel (`qtip_bytecut_design §4.2`); (b) the trellis adds a **serial state dependence** (`state[i] ← state[i-1]`) with **no Q4_K analog** (Q4_K nibbles are independent), which the 32-wide simdgroup cannot amortize unless the encoder emits lane-independent sub-blocks. Both place the trellis on the **wrong side of the same compute-bound wall** that killed Q3_K. At 24.2% peak vs the ~55% GO bar the proxy is decisive: predec already *removed* per-element work and Q3 still sat compute-bound at 24%; a trellis adding more per-element work in the same place has no path to BW-bound. The direct kernel (`gemm_qtip_trellis_v1`, 4.1-1) was **deliberately NOT built** per the scout's "cheapest honest outcome" — the proxy suffices to record Type-1.
**Type:** **Type-1** — a measured property of per-element decode-ALU density on this GPU (M3 Pro, unified memory). The Q3_K proxy is the direct precursor; the trellis inherits its death by the same mechanism and adds a second structural hazard (serial state). Neither is an implementation weakness.
**Type-2 reframe (named, alive behind its cheap oracle):** **lane-independent sub-block layout with fused predec-of-trellis-seeds** — if a co-designed Colab encoder emits 32 lane-parallel trellis segments per block AND a partial trellis-seed predecode table hoists the per-seed ALU out of the inner loop, the per-lane compute-per-bit could narrow below Q3_K and the 96–104 B block might flip BW-bound. **Cheap oracle (named, in-hand):** build `gemm_qtip_trellis_v1` in the lane-independent layout (`quant.metal`, clone `gemm_q4_k_v4_predec_2r` outer at :2358, inner = 1MAD trellis walk on 32 independent per-lane segments), bench via `tests/q3k_bytecut_bench.rs:bench_shape` on all 3 Qwen shapes; GO floor ≥55% peak AND beats Q4_K-predec µs/call. Resurrect ONLY if the encoder can provably emit the lane-independent layout — if it cannot, the serial-state hazard is unretirable and the reframe is dead before the kernel.
**Axis closure:** recording this CLOSES the sub-Q4 byte-cut axis for decode speed — all four formulations are now dead: (1) **Q3_K direct** (Type-1, 22–24% peak, clean), (2) **low-rank + compressible residual codec** (Type-1, data-aware reframe ran and died), (3) **learned per-model codebook** (Type-1, Apple no-hardware-gather), (4) **QTIP-on-Metal trellis decode** (Type-1 by proxy, this entry). Dense-tps headroom routes to **dispatch fusion (2.2) / spec / stateful ONLY, not bytes.** The footprint lever (~1.2 GB QTIP vs ~1.9 GB Q4_K on disk) is theoretically real but gated behind the QTIP **quality** oracle, which is itself leaning NO-GO (`QTIP 3-bit trellis vs Q4_K_M`, `bits_needed=[+1.37,+0.44]`).
**Killing memory:** [[moat-status-forward-path-2026-05-31]]; the clean-room Q3 proxy in the **Q3_K sub-Q4 decode byte-cut** entry (36.3 GB/s = 24.2% peak, re-confirmed 2026-06-01) + `reports/autonomous_run_2026_06_01.md`; `scout_phase_4_1_qtip_spike.md` §4.1-0.
**Resurrection check:** do NOT build `gemm_qtip_trellis_v1` to re-test this Type-1. The ONLY resurrection path is the named Type-2 oracle above, and ONLY if a Colab encoder demonstrably emits the lane-independent layout. Do not resurrect on the per-token compute-headroom argument — the **Q3_K sub-Q4 decode byte-cut** entry is the standing counterexample that per-token headroom ≠ per-kernel BW-bound. The QTIP **quality** arm remains a separately-open Type-2 (Colab codec not run), but a quality GO is moot while the decode cost is Type-1 dead.

---

## 🪦 f16 activations — f16-x into the predec GEMV (Phase 2.1-b)

**Status:** dead-for-tps 2026-06-02 — **Type-1 (measured)**. Closes the activation-precision axis for the GEMV.
**Charge:** halve the x-activation traffic into the dominant predec GEMV (load x as f16 instead of f32) — the one in-kernel BW sub-lever the 2.1 scout left open, gated on a microbench showing x-traffic is a measurable GEMV sub-term *after* f16-scales.
**Evidence (oracle, Wave-4c f16x-probe):** built `gemm_q4_k_v4_predec_pair_f16x` — an exact clone of the shipped f16-scales fused gate+up GEMV (`_pair_f16s`) with the ONLY change being the x stream loaded as `half` (2 B) and widened to float in-register so the FMA chain is byte-identical — plus a paired bench vs `_pair_f16s`. On the dominant **11008×2048** FFN gate+up shape: **f16x 519.685 µs vs f16s 519.303 µs = −0.07% (noise); GB/s 59.8 vs 59.9.** The bench prints the root cause: **x is 0.0263% of the f16s read traffic; halving it removes 0.0132% of bytes.** The activation is re-streamed per output tile but is dwarfed by the weight (1152 B/block-row) + f16 scale streams — the GEMV is weight+scale-bound, not x-bound.
**Type:** **Type-1** — a measured property of the GEMV byte budget (x ≈ 0.026% of traffic at 11008×2048). Confirms the **Decode-kernel micro-opt A5/A6/A10** reconciliation ("a half-precision x stream is killed by byte arithmetic; x ≈ 0.03% of `_pair` traffic"). Halving 0.026% of bytes cannot move the bus — not an impl weakness.
**Resurrection check:** do NOT re-test on the predec GEMV. The only regime where x-traffic could matter is cols ≫ weight bytes (a very wide, very short matrix) — which the Qwen decode GEMVs are not. The probe kernel + bench were **reverted** (the measurement lives here; a structural 0.026% won't change with kernel cleverness, so no in-tree oracle is kept). With the byte-cut axis (QTIP) and now the activation-precision axis (f16-x) both Type-1 dead, the GEMV is at its byte optimum. The live activation-precision lever is **f16-KV** (the *cache*, not x) — shipped (`ed6925e`/`1fa6941`).
**Killing memory:** [[overnight-haul-2026-05-31]] (the A5/A6/A10 reconciliation that predicted this).

---

## 🪦 Q4_K batched MMA (simdgroup-matrix) on rows ≤ cols shapes

**Status:** killed 2026-05-31 by paired microbench (Type-1 occupancy) — PARTIAL kill; the rows>cols variant is GO (shape-gated, integration deferred — see handoff)
**Evidence:** P1 prefill-MMA (`gemm_q4_k_m_batched_v3w_mma`, one-simdgroup/8-rows tile). Paired N=8 GEMM microbench vs v3w, parity-green (atol 8e-5→1.26e-4 fp16, token-identical): tall ffn gate/up (11008×2048, rows>cols) = **+22–24%** WIN; ffn_down (2048×11008) = **−8.8%**; attn q/o (2048×2048) = **−10–16%** LOSE. On rows≤cols the ceil(rows/8)-threadgroup geometry underfills the M3 Pro — a measured hardware-occupancy property, not an impl weakness. So MMA is shape-gated to rows>cols.
**Type:** Type-1 for rows≤cols (occupancy reality — square/wide attn + ffn_down). Type-2 reframe: a **multi-simdgroup-per-TG** tile would fill on small-rows shapes — alive only behind an offline occupancy oracle (TG count vs M3 Pro core count at those dims). Separately, the GO tall-shape MMA is **dormant in the shipped (predec-on) batched path** and needs a **predec-MMA twin** + batched predec scale-table coverage to fire.
**Killing memory:** [[moat-status-forward-path-2026-05-31]]; handoff `plans/p1_prefill_mma_integration_handoff_2026_05_31.md`; branch `worktree-agent-a08c1cb44eb3d4e47` (`c9b1c07`).
**Resurrection check:** do NOT wire MMA into attn/ffn_down (rows≤cols) without the multi-simdgroup tile + its occupancy oracle. The tall-shape MMA is GO but needs the predec-MMA twin to help the shipped path (which uses predec, not v3w).

---

## 🪦 Q5_0 simd_shuffle byte broadcast (audit fix #1)

**Status:** killed 2026-05-14 by -3.5% regression
**Evidence:** Bit-identical 3-token parity but benches -3.5% trimmed_mean. Apple's HW already coalesces redundant lane-pair byte loads; simd_shuffle overhead exceeds the savings.
**Killing memory:** [[feedback-kernel-parity-gate]]
**Resurrection check:** if Apple silicon coalescing changes (new GPU gen), retry; otherwise the HW already does this.

---

## 🪦 Decode-kernel micro-opt: vectorized uint4 unpack (A5) + threadgroup/occupancy tuning (A6)

**Status:** killed 2026-05-31 — **Type-1** (both), NO-CHANGE, reverted clean. Part of the overnight kernel haul that closed the decode-GEMV micro-opt track (closeout `3cb5944`; profile `f2a6a4f`).
**Type-1 or Type-2:** Type-1 (Apple-GPU memory-model facts, not impl weakness).
**Evidence:** `plans/overnight_build_queue_2026_05_31.md` §A (A5, A6). **A5 (vectorized `uint4` nibble unpack on `_pair`):** the predec GEMV loads are *already* simdgroup-coalesced, so a wider `uint4` load buys no bandwidth AND cannot apply without reordering the bit-identical FMA chain (would break greedy parity). The stall is occupancy / scale-read / x-traffic, not load width. No commit. **A6 (threadgroup / occupancy tuning):** `_pair` is already oversubscribed (~76 TGs/core) so there is no occupancy lever; threadgroup-size sweeps were noise (tg384 −0.2%, below gate). Reverted clean. The A4 profile (`f2a6a4f`) localizes the stall: `predec_pair` is 46.6% of decode at ~56% of peak BW — the gap is scale-byte volume (addressed by the A6.5 f16-scales win, `0899137`) + layout (A10, also Type-1 dead), NOT load width or geometry.
**Killing memory:** [[overnight-haul-2026-05-31]]; sibling decode-kernel kills A7 (Q4_K batched MMA) + A10 (access-order layout), this section.
**Resurrection check:** do NOT re-test (Type-1). Arc conclusion: the Q4_K predec decode GEMV is at the Apple-GPU memory-model optimum for batch=1 (M=1) decode. Remaining dense-tps headroom is fewer bytes (QTIP) or the spec / stateful axes, NOT decode-kernel micro-opt. A6.5 (f16-scales, `0899137`, opt-in) was the lone bandwidth win of the track.
**Update 2026-06-01 (paradigm Phase 1→2 boundary — reconciliation, does NOT resurrect):** the fresh 0.2 gap-anatomy (`paradigm_execution_log.md`) measured the predec GEMV at ~52% peak / a 1.55–1.63× llama gap and noted llama advertises `simdgroup matrix mul = true` + `use residency sets = true`. A dedicated adversarial reconciliation (`reports/scout_phase_2_gemv_reconciliation.md`) confirmed this does **NOT** reopen A5/A6/A10: (i) ggml-metal uses `mul_mv` (vec) at M=1 and `mul_mm` (MMA) only for batched/prefill — same split dismantle made, so llama's MMA flag is **not** a decode-GEMV lever (cf. the "Q4_K batched MMA rows≤cols" Type-1 kill); (ii) `use residency sets` is Type-1 dead on unified memory (silicon #3); (iii) the one genuinely-distinct unscoped form — a half-precision **x** activation stream into the predec GEMV — is killed by byte arithmetic (x ≈ 0.03% of `_pair` traffic; the f16s bench `q4k_predec_f16s_bench.rs:125` already counts it). **The 1.55× gap is bytes (QTIP, Phase 4) + dispatch count (2.2), NOT a GEMV-kernel micro-opt.** `scout_phase_2_1_gemv.md` (which re-proposed A5-a/A5-b/A6) is **SUPERSEDED** by the reconciliation; do not execute its A5/A6/A10 kernel work. The cheap oracle for any future GEMV-BW claim is the existing `cargo test -p dismantle-core --test q4k_predec_f16s_bench -- --nocapture` (GB/s readout); a form that doesn't beat the f16s readout at 11008×2048 is dead before integration.

---

## 🪦 A10 access-order weight-layout repack (Q4_K predec GEMV)

**Status:** killed 2026-05-31 — **Type-1**, built + measured + reverted (tree clean at HEAD).
**Type-1 or Type-2:** Type-1 (Apple-GPU memory-model fact).
**Evidence:** Built `repack_q4_k_pair_access_order` (permute the 128-B qs plane so each thread's 4 nibble bytes are contiguous at `16+lane*4+pi` instead of stride-32 `16+pi*32+lane`) + matching `gemm_q4_k_v4_predec_pair_ao` + parity/bench (`q4k_ao_repack_bench.rs`). On the dominant 11008×2048 FFN gate+up shape: the **bit-identical** scalar-load variant runs **−16.8%** (49.7 vs 58.0 GB/s) — per-thread contiguity de-coalesces the simdgroup (32 lanes per `pi` now span the whole plane instead of 32 contiguous bytes = 1 transaction). The stride-32 original is *already* the optimally-coalesced layout (confirms A5).
**Reframe considered:** vectorized `uint` 4-byte load on the repacked layout (the only formulation that uses the contiguity). Dies twice: (a) NOT bit-identical — ~1 ULP FMA-recontraction drift (−71.67029 vs −71.67032) → fails the A10 hard gate; (b) no BW gain (48 GB/s < 58-64 GB/s baseline; 5-run sweep {−32,+16,+28,−23,+19}% = pure Claude.app GPU contamination, no signal). No formulation gets a wider per-thread transaction while keeping BOTH bit-identity AND simdgroup coalescing.
**Killing memory:** [[a4-per-kernel-decode-profile]], A5/A6 (BW-bound, loads already coalesced).
**Design note:** `reports/a10_layout_repack_design.md`.
**Resurrection check:** do NOT re-test (Type-1). The live BW levers are scale-byte volume (A6.5 f16-scales, shipped) and lower weight precision (A8 Q3_K, footprint-only until a Q3_K predec/2r rewrite) — separate non-bit-identical levers, each needs its own quality oracle, NOT a resurrection of this kill.

---

## 🪦 Q8-KV layer-differential precision

**Status:** killed 2026-05-21 by uniform routing
**Evidence:** Per-layer routing balance 0.987-0.995 means there's no signal driving "which layers' KV needs higher precision". The intuition was that layers with concentrated routing might tolerate lower-precision KV; calibration shows no concentration exists.
**Killing memory:** [[corpus-complete-analysis-landed]]
**Resurrection check:** if a new model shows skewed routing (balance < 0.95), retry. Current uniform-Q8 stays per [[q8-kv-landed]].

---

## 🪦 Semantic cache (Bible §8.1 L1.2 extension) — PARKED, not reality-dead

**Status:** NO-GO 2026-05-31 on the git-history proxy — **Type-2 (parked behind a named, built oracle)**
**Evidence:** `reports/oracle/semantic_uplift.json` + `reports/oracle_semantic_uplift.md` (`oracle_prefix_cache.py` incremental-reuse mode). Incremental reuse OVER the shipped default-on exact prefix cache = **+1.48 pts mean / +0.00 median / +13.2 max** across 14 sessions, vs the ~10-pt gate. 12/14 sessions = +0.00 (the exact tier already harvests every consecutive shared prefix); retrieval precise (100% verify-confirm at τ_sem=0.80, MIN_REUSE=16). The kill is opportunity, not recall.
**Type:** Type-2. The mechanism provably works (2 return-to-prior-file sessions at +13.2/+7.5); it died on the proxy's consecutive-edit *workload shape*, not on reality. The +0.00 median is also positive evidence the shipped exact prefix cache is doing its job.
**Killing memory:** [[moat-status-forward-path-2026-05-31]]
**Resurrection check:** re-run the SAME oracle (`oracle_prefix_cache.py`, ~13 s, no GPU) on REAL file-interleaved session logs. GO there ⇒ build `InMemorySemanticIndex` per `plans/stateful_moat_continuation_design_2026_05_31.md` §1.5 (build plan executes unchanged). Do NOT build on the proxy number; do NOT bury as reality-dead.

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

## 🪦 Usage-frequency vocab screen w/ norm-bound certificate (Bible §8.1 L3.1)

**Status:** NO-GO 2026-05-31 by offline oracle — **Type-1 (reality-dead)**
**Evidence:** `reports/oracle/vocab_coverage.json` + `reports/oracle_vocab_coverage.md` (`oracle_vocab_coverage.py`, real GGUF lm_head dequant). Certified-fast-path rate = **0%** across the full ‖h‖/ℓ_c sweep and across H=256→32768. The norm-bound certificate (out-of-H token v provably-not-argmax iff ‖w_v‖·‖h‖ < ℓ_c) needs cos(w_c,h) > **1.0–1.46** to fire — unreachable (cos ≤ 1). Coverage is fine (H=7,119 covers 99.9% of occurrences) — NOT the blocker. Smoking gun: the 10 highest-norm lm_head rows are RARE tokens (corpus freq 0, freq-rank 22k–146k), so a frequency hot set never includes them → max out-of-H norm pinned at the global max → the Cauchy-Schwarz bound is structurally too loose.
**Type:** Type-1 — a measured head property (similar row norms, cond≈45 full-rank, + norm/frequency anti-correlation). Distinct mechanism from the dead SVD screen, dies the same way. lm_head is only ~4–10% of bytes/token → small ceiling regardless.
**Type-2 reframes (dead-until-their-oracle, NOT resurrected on vibes):** block-max / per-coordinate certificate (tighter than scalar Cauchy-Schwarz); data-aware real-argmax hot set (capture the true argmax stream via `usage_capture` instead of input-token frequency). Each alive only behind its own cheap offline oracle.
**Killing memory:** [[moat-status-forward-path-2026-05-31]]; sibling SVD-screen kill [[kill-protocol-reframe-audit-2026-05-30]].
**Resurrection check:** do NOT build the scalar-norm-bound screen. A reframe ships only after its named oracle clears (a tighter certificate certifies ≥80%, OR the data-aware hot set with REAL argmax frequencies changes the coverage picture).

---

## 🧱 Silicon-architecture custom solutions — full build/bench/audit session (2026-05-29)

**Provenance:** 16 candidate Apple-Silicon levers, each built into real code
(build → bench → audit-to-improve) in the untracked `silicon-builds/` tree
(629 MB, zero commits). The per-solution `VERDICT.md` files were **transcribed
into this ledger on 2026-06-01 immediately before `silicon-builds/` was deleted**
(untracked → git cannot recover; the two LIVE levers were first re-derived
parity-green in `main`). This section IS the surviving record. Full prose lived
in `silicon-builds/<crate>/VERDICT.md` + `silicon-builds/SUMMARY.md`; the
canonical strategy crosswalk is `plans/bible_archive.md` (axes 1–3).

**Scoreboard (16 solutions across 15 crates; #4+#9 share `dispatch`, #16+#17 share `mixedprec`):**

| # | Solution | Verdict | Type | Headline |
|--:|----------|---------|------|----------|
| 8  | Q4_K simdgroup-MMA kernel | ✅ **LIVE → SHIPPED** | — | +10–20% batched Q4_K GEMM; shape-gated rows>cols, shipped `14434f5` (`DISMANTLE_QWEN_Q4K_MMA`) |
| 13 | Zero-copy mmap MTLBuffer loader | ✅ **LIVE → SHIPPED** | — | 1673× faster bind, −1.9 GB RSS, −324 ms TTFT, bit-identical; shipped `a08fa17` |
| 16 | Mixed-precision weights (Q3_K/Q2_K) | 🟢 PRIZE→AWQ | T1 (naive) | naive RTN dead (uniform sensitivity); quantifies +31%/+71% BW prize claimable only by smart quant |
| 15 | int4 KV cache (per-channel) | 🟡 HELD (quality-gated) | — | cosine 0.998, 1.2 GB→340 MB @32K; long-context memory enabler, needs PPL gate |
| 7  | GPU top-K sampler | ⏸ DEFERRED | — | greedy argmax already on-GPU; only temp>0 gap, unbenched — not built |
| 1  | Hybrid AMX+GPU GEMM | ❌ DEAD | T1 | fair batched GPU beats AMX everywhere; AMX can't eat Q4_K |
| 2  | Super-page mempool | ❌ DEAD×2 | T1 | arm64 refuses 2 MB pages (kr=4); sequential streaming already TLB-amortized 144× |
| 3  | MTLHeap residency v2 | ❌ DEAD | T1 | heap 8.5% slower than separate shared buffers even batched |
| 4  | ICB single-pass forward | ❌ DEAD | T1 | host encode = 0.27% of per-dispatch budget + already overlaps |
| 5  | Non-GEMM CPU offload | ❌ DEAD | T1 | non-GEMM = 3.2% of wall + on serial dependency chain |
| 6  | ANE / CoreML FFN | ❌ DEAD (measured) | T1 | ANE FFN bandwidth-bound 56 GB/s, ~4–7× slower than Q4_K GPU FFN; can't eat Q4_K |
| 9  | Argbuf pre-compiler | ❌ DEAD | T1 | same as #4: host encode negligible |
| 10 | Weight prefetcher | ❌ DEAD | T1 | WILLNEED −29% once warm; weights always warm in 18 GB |
| 11 | Q4_K+AMX fused decoder | ❌ DEAD | T1 | cblas can't ingest Q4_K; AMX loses on GFLOPS ceiling (#1) |
| 12 | mlock allocator | ❌ DEAD (coexist) | T1 | works but pins 10% RAM from slm — inverts coexistence contract |
| 14 | Multi-command-queue | ❌ DEAD (prod) | T1 | GPU saturated per decode kernel (32–1376 TGs ≫ 16-TG crossover) |

**Tally: 2 LIVE+SHIPPED · 1 prize-redirect · 1 held · 1 deferred · 11 dead.**

The convergent conclusion (M3 Pro shared-bus unified memory): the only live decode
levers are **GPU-side kernel work (#8)** and **load-path memory (#13)**. Every
cross-engine lever (#1 AMX, #5 CPU, #6 ANE, #11 Q4_K+AMX) and every
memory-placement / host-dispatch lever (#2, #3, #4, #9, #10, #12, #14) is dead:
no second engine helps the bus-bound dependency-chained decode; the weights are
already optimally placed (RAM, shared, warm); the GPU is saturated per decode
kernel so host dispatch is off the critical path. The single biggest number on
the board is the low-bit bandwidth prize (#16: +31% all-Q3_K / +71% all-Q2_K) —
real, but RTN can't claim it and the 2026-06-01 Colab gate (below) found the
requant-from-Q8 forms NO-GO.

### Per-solution kill protocol (the 11 dead + #16-naive)

- **#1 Hybrid AMX+GPU GEMM — Type-1.** Fair batched f16 GPU GEMM beats `cblas` AMX on FFN/LM-head/most B≥8 cells; AMX bandwidth-bound ~50 GFLOPS at these shapes. Production weights are Q4_K (4× less bus traffic) and AMX **cannot consume Q4_K** → real gap is wider in the GPU's favor. *Reframe:* the "independent engines" surprise (concurrent overlap_eff 0.81–1.13) redirects to **non-GEMM CPU offload** — that's #5, separately dead. *Kept:* the concurrent probe (canonical "does HW overlap X with Y" test).
- **#2 Super-page mempool — Type-1 ×2.** (a) `mach_vm_allocate` returns `kr=4` for `VM_FLAGS_SUPERPAGE_SIZE_2MB` — the syscall is x86_64-only, closed on arm64. (b) Even if it worked: Q4_K GEMV streams **sequentially** (78.4 GB/s, 0.816 ns/line), amortizing one page-walk over 256 cache lines — dTLB already a non-factor (144× faster/access than the random pattern bigger pages target). *Reframe:* a random-access structure (large hash-indexed KV store) — none exists. Allocator kept only against that hypothetical.
- **#3 MTLHeap residency v2 — Type-1.** A single batched-`useResources` MTLHeap is **8.5% slower** (84.8 vs 92.7 GB/s) than separate `StorageModeShared` buffers. On unified memory the weights live in RAM and the GPU reads directly — there is no residency to optimize; heap residency-tracking (designed for discrete-GPU Private/Managed storage) is pure overhead. *Reframe:* none on Apple Silicon. `load_heap_resident` infra stays held-but-unused.
- **#4 ICB single-pass forward + #9 argbuf pre-compiler — Type-1.** Host-side per-dispatch encode = **0.402 µs/dispatch = 0.27%** of the 147 µs/dispatch GPU budget, and it already overlaps (CPU encodes N+1 while GPU runs N). Same family as the bible **host-side per-dispatch overhead** kill (ICB, concurrent encoder, PSO batching — all cap below ship gate). *Reframe:* ICB could simplify a CPU-bound prefill with thousands of tiny dispatches — not the decode lever proposed.
- **#5 Non-GEMM CPU P-cluster offload — Type-1.** Decode non-GEMM (rmsnorm/rope/softmax/silu) = **3.2% of wall** AND no dependency-free slot (each op feeds the next layer's GEMM; can't run concurrent with the same layer's GEMM; cross-layer overlap is pipelining, already dead). *Reframe:* a fully-parallel CPU-side speculative draft (the eagle5 case, not non-GEMM offload) — separately NO-GO (see EAGLE-3 entry).
- **#6 ANE / CoreML FFN — Type-1 (measured).** Built + ran the Qwen FFN on the ANE: latency batch-independent (~2.4 ms @ B=1/8/64) ⇒ **weight-bandwidth-bound at 56 GB/s** (135 MB fp16 / 2391 µs), same ceiling AMX hit. The GPU Q4_K FFN reads ~34 MB (4-bit) at higher effective BW → ANE is **4–7× slower**, and the ANE **cannot consume Q4_K**. Closes the cross-engine front with hard numbers. *Reframe:* a static-shape fp16 side-model the GPU is too busy for — a different problem, no current need.
- **#10 Weight prefetcher — Type-1.** Single-arm probe lied (cold 62 ms vs WILLNEED 2.1 ms → "96.6% hidden"); the 3-arm decomposition shows that 62 ms is the **one-time SSD read**, after which the 1.8 GB model lives in the macOS buffer cache (≪18 GB) forever, and bulk `WILLNEED` on a warm buffer is **−28.8%** (syscall + readahead bookkeeping > touching warm pages). `MADV_DONTNEED` can't force eviction with free RAM. *Reframe:* prevent eviction via mlock (#12) — also dead.
- **#11 Q4_K+AMX fused decoder — Type-1.** Three blockers: (a) `cblas` requires a materialized f32/f16 matrix — you cannot inject per-block Q4_K dequant into the BLAS inner loop, so AMX needs a full extra dequant pass + 4× memory, forfeiting the 4-bit advantage; (b) even with f32 already materialized AMX loses (#1's measured 45.5 vs 112.4 GFLOPS LM-head); (c) hand-rolled AMX intrinsics with inline dequant = multi-week and still hits the GFLOPS ceiling. The ceiling, not the dequant, is the wall. *Home for "matrix HW on Q4_K":* #8 GPU simdgroup-MMA, which keeps the 4-bit bandwidth.
- **#12 mlock allocator — Type-1 (for coexistence).** mlock works (wires 1.93 GB in ~0.1 s) but gives **zero steady-state speedup** (it changes eviction policy, not access speed; weights are never evicted in 18 GB anyway) and **inverts the coexistence contract** — it pins 10% of RAM away from slm exactly when slm needs it, the opposite of the `coexist.sh` yield model. *Reframe (alive as a documented option, not a default):* solo low-latency serving with strict tail-latency SLAs → opt-in `--mlock-weights`. A deployment-mode choice, never a decode tps lever.
- **#14 Multi-command-queue — Type-1 (in production).** Audit-to-improve: aliased output buffers gave a false "GPU serializes, DEAD"; disjoint buffers reversed it to a real 1.37× — **for kernels that underfill the GPU**. The grid sweep finds the crossover at **~16 TGs** (M3 Pro ≈ 18 cores); every production decode kernel dispatches **≥32 TGs** (k/v 32 · q/o 256 · gate/up 1376 · LM-head 4K–19K), all ≥2× past saturation, so a concurrent sibling finds no idle cores. Quantitatively explains the held `qkv_concurrent` +1.68% (kernel-tail occupancy dip only). *Reframe:* none in decode — the mechanism is live but has no underfilled kernel to apply to.
- **#16 Mixed-precision weights, naive per-family — Type-1.** Re-quantizing the real Qwen-3B Q4_K_M tensors with an MSE-optimizing quantizer: sensitivity is **uniform** (every family rel_L2 ≈ 0.14 @3b, ≈ 0.32 @2b — 3% spread → no tolerant family to cherry-pick) AND the model is **already** Q4_K_M-mixed (attn_v/ffn_down/token_embd pre-upgraded to Q6_K). Naive RTN-3b adds ~14% L2/tensor — far past the W4A8 logit-cosine-0.999 budget. *Reframe (the live byte-cut path):* **importance-aware quant (AWQ/GPTQ, #17)** protects salient channels — RTN-3b 0.14 → AWQ-3b 0.07–0.10 on the real weights; the +31% (Q3_K) / +71% (Q2_K) prize is the largest single lever on the board, claimable only by smart quant. *Oracle:* the model-level logit/PPL gate — **run 2026-06-01 (below), NO-GO for the requant-from-Q8 forms.**

### Not kills (recorded for completeness)
- **#7 GPU top-K sampler — DEFERRED, not built.** Greedy argmax is already GPU-side (`sample_argmax_f32`, folded into the TCB, 4 bytes cross the bus); every gate/bench is greedy temp=0. A fused `topk_logits_f32` is a real gap for temp>0 (today reads the full 152K-elem f32 logit vector to the CPU) but no bench exercises temp>0 → building it now is unvalidatable speculation. *Build trigger:* dismantle ships temperature sampling + a sampled-generation bench.
- **#15 int4 KV cache — HELD, quality-gated.** Per-channel int4 is the clear best scheme (cosine 0.998, rel_L2 ~0.06/head/layer) and a strong **long-context memory enabler** (1.2 GB→340 MB @32K, 3.5×) — but NOT a short-context tps lever (attention ~2.4% of decode) and per-layer error compounds over 36 layers. Hold pending a real-model perplexity/token gate (same rig as W4A8); ship as opt-in `--int4-kv` long-context mode only if it passes. Cross-ref `[[q8-kv-runtime-landed]]` (Q8 KV shipped; payoff only at long context).
- **#8 / #13 — LIVE, now SHIPPED in `main`.** #8 Q4_K simdgroup-MMA: shipped shape-gated to rows>cols (ffn gate/up) at `14434f5`; the rows≤cols variant is a Type-1 occupancy kill — see the **Q4_K batched MMA on rows ≤ cols shapes** entry above. #13 zero-copy loader: shipped at `a08fa17` (−1.9 GB RSS, −324 ms TTFT, bit-identical A/B).

---

## 🪦 sub-Q4 byte-cut quality gates — Colab verdicts (2026-06-01)

The two decisive Colab quality gates for the sub-Q4 byte-cut axis (run by the user
on an RTX PRO 6000 Blackwell, 102 GB, master notebook
`colab/05_combined_quality_gates.ipynb`; raw JSON at
`reports/colab_verdicts/{imatrix_mixprec_results.json, qtip_3bit_results.json}`).
**Net: the entire sub-Q4 byte-cut axis (trellis + weight-mixing) has NO live bet;
no GO cleared any gate → no default flips.**

### 🪦 imatrix mixed-precision (non-uniform Q4/Q3 allocation) — Type-1

**Status:** NO-GO 2026-06-01, **Type-1**. Closes the imatrix arm of #16's AWQ redirect for the requant-from-Q8 form.
**Evidence:** `imatrix_mixprec_results.json`. llama-imatrix on the q8≈f16 source → uniform Q4_K_M (gold) vs the intended attn+ffn_gate@Q4_K / ffn_down+ffn_up@Q3_K mixed GGUF. The `--tensor-type` Q4/Q3 override **would not take in this llama.cpp build** (`AssertionError('mixed GGUF not produced')`), so the decisive logit leg ran against the **uniform Q3_K_M + imatrix** fallback: 15% smaller (1.66 vs 1.96 GiB) but **worse on every quality axis** — PPL 4.68 > 4.59 (gold), logit cos 0.983 < 0.993, KL 0.053 > 0.022, argmax 0.911 < 0.922 (192 tokens, code calib).
**Reframe considered:** the *true* ~3.82-bit non-uniform Q4/Q3 split (the leg the build couldn't produce), decisively logit-tested. **Why it also dies → its oracle already fired:** the local weight-RMSE oracle ([[sub4bit_quant_oracle_2026_05_29]], silicon #16) already showed the mixed allocation trails Q4_K on **7/7 tensor families** (uniform sensitivity, no tolerant family) — non-uniform allocation cannot beat uniform Q4_K at equal-or-fewer bytes. The form not tested is bounded by an oracle that says it loses → Type-1.
**Resurrection check:** do NOT re-test imatrix mixed-prec as a requant-from-Q4K/Q8 byte-cut. The only surviving smart-quant path is **AWQ-from-f16** (importance-aware, not requant-from-already-quantized) — a different source regime (`DISMANTLE_QWEN_AWQ`, still HELD at 1.115×, [[awq-option-b-local-green-2026-05-26]]); the requant-from-quantized forms are dead.

### 🪦 QTIP 3-bit trellis vs Q4_K_M — leaning NO-GO (Type-2 open: decisive codec not run)

**Status:** NEEDS-MEASUREMENT / **leaning NO-GO** 2026-06-01. **Not yet a recorded decisive kill** — per the Kill Protocol the weight-bracket alone cannot close a Type-2.
**Evidence:** `qtip_3bit_results.json`. LEG1 GGUF↔HF transpose fix verified (q4k_rmse ~0.07 Q4_K / ~0.018 Q6_K — sane). The corrected weight-quality bracket `bits_needed = [+1.37, +0.44]` is **both positive** ⇒ even best-case QTIP-3bit is ~0.44 bits short of Q4_K_M **weight** quality (agrees with the local proxy ~1.20, [[qtip-quality-oracle-2026-05-31]]). The decisive real Cornell-RelaxML RHT+trellis codec + logit/PPL legs were **deliberately NOT run** (`ALLOW_FRESH_QTIP_CODEC=False`, a conscious compute save) → `legs.{weight_rmse_le0, logit_ge_q4k, ppl_le_gate}` all `null`.
**Type:** Type-2 (open). The mechanism (arithmetic-coded bitshift trellis, gather-free, no hmask residual) is a genuinely different formulation from the dead Q3_K decode byte-cut and the dead low-rank/codebook levers; it is the **single** surviving sub-Q4 byte-cut bet. But the measured bracket is a property of the *target* weight quality, not of QTIP's codec — so it leans NO-GO without **recording** a kill.
**Resurrection / decisive-kill check (named oracle, in-hand):** re-run `colab/05_combined_quality_gates.ipynb` with `ALLOW_FRESH_QTIP_CODEC=True` (~20–40 min, Drive-resumable). A GO needs the real codec to clear weight-RMSE ≤ 0 AND logit ≥ Q4_K AND PPL ≤ gate on code; given both bracket bounds are positive and the proxy agrees, the expected outcome is a decisive **Type-1** kill — but it is not recorded as one until that codec runs. Cross-ref the **Q3_K sub-Q4 decode byte-cut** entry (QTIP is its live Type-2 reframe). **Decode-cost cross-ref (2026-06-02):** the Phase 4.1 *decode* gate is now recorded **Type-1 by proxy** (see **QTIP-on-Metal trellis decode** above) — so a quality GO here is **necessary but NOT sufficient** to reopen the byte-cut axis; the decode Type-2 (lane-independent sub-block) must also clear its oracle first. Do not run the Colab quality codec to reopen a dead decode path.

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
- [[silicon-solutions-2026-05-29]] — the 16-solution silicon audit (2 LIVE→shipped, 11 dead); source of the Silicon-architecture kills section
- [[sub4bit-quant-oracle-2026-05-29]], [[qtip-quality-oracle-2026-05-31]] — the weight-RMSE + QTIP proxy oracles behind the 2026-06-01 Colab verdicts
- [[moat-status-forward-path-2026-05-31]] — post-sweep moat: prefix-cache + draft-tuning live; sub-Q4 byte-cut routes through QTIP only (now leaning NO-GO)
