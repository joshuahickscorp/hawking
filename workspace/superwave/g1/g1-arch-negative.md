# G1 architecture negative ledger

Authority for this file: sweep of `HEAD` at `2eee9a004` via `git show` (sparse checkout; missing-on-disk is not missing-in-git). No GPU, no generate, no artifact write. Every number below is copied from a named receipt or condensed kill-table. This lane did not re-measure.

Labels: **MEASURED** = receipt names a GPU/wall/generate protocol. **PROJECTED** = arithmetic on a measured quantity. **ESTIMATED** = model, not a run. **RECORDED** = condensed kill-table with no primary receipt located in this sweep. **CLAIMED** = campaign figure this lane did not independently verify.

A roof is conditioned on the execution genome that produced it. Several kills below died under a serial-extract / expand-then-GEMV / uniform-Q4 genome. G1 intends to change that genome. The reopen column is the gate, not a suggestion.

Binding (campaign law, restated): reject any low-BPW path whose production shape is `low-bpw → expand to float/Q4 → generic GEMV` unless a complete-token measurement proves the expansion is still a net physical win. Preferred shape is low-BPW codes consumed in-register by a representation-specific Metal kernel.

G0 regime (CLAIMED in the G1 brief; the nearest on-disk receipt is `RUNG_QWEN38_MEASURED.json`, not an independent remeasure):

| quantity | brief CLAIMED | nearest receipt field | label |
|---|---|---|---|
| complete BPW | ~4.2527 | `QWEN38_TOKEN_NS_LEDGER.json` `bpw` = 4.252735126866492 | MEASURED in that receipt |
| TOKEN_NS | ~37,900,000 | `RUNG_QWEN38_MEASURED.json` `ns_per_token` = 38216792.0 | MEASURED in that receipt |
| TPS | ~26.4 | same file `rung.tps` = 26.166508167404526 | MEASURED in that receipt |

Do not treat those three as this lane's baseline. They are the incumbent record G1 is trying to beat.

---

## Compact table

Columns: mechanism · verdict · kill number · receipt · reopen.

| # | mechanism | verdict | kill number | receipt | reopen |
|---|---|---|---|---|---|
| 1 | Full-layer / MoE megakernel as the tps lever | KILLS on two independent grounds. (A) MoE megakernel-deeper: remaining hand-off is 0.04% wall (RECORDED). (B) Qwen-3B megakernel is a pass-through POC, not a measured token path. (C) fused expert-wave collapse is a measured 71% regression. | (A) 0.04% remaining wall. (C) MEASURED 1413 vs 828 ms/token = 0.59× / +71%. | (A) `workspace/docs/guides/dead_levers.md:25`. (B) `crates/hawking-core/src/kernels/megakernel.rs:1-8`, shader `megakernel_qwen3b.metal` header. (C) `workspace/campaign/evidence/systems/hawking/HAWKING_EXPERT_WAVE_NEGATIVE.json` `measured`. Related: NS-020 CB collapse, NS-030 single-dispatch `L@(R@x)` lost 5–13×. | (A) new cross-TG sync primitive. (B) not a kill to reopen — it was never a complete kernel. (C) profile the fused wave and show it does not serialize previously overlapped matvecs; then a warm paired complete-token A/B. Do not retry “fewer drains ⇒ faster token”. |
| 2 | GEMV / kernel-micro-opt axis on Qwen3.8 | CLOSED on the current unique-once Q4 GEMV genome. Qwen3.8 is at the honest decode ceiling. | MEASURED 406.2 GB/s vs honest unique-once ceiling 411.51 GB/s = 98.7%. Rung record 36.987 ms GPU / 38.217 ms wall / 26.17 tok/s. Decode micro-opt A5/A6: NO-CHANGE, “GEMV at memory-model optimum” (RECORDED). f16-x into predec GEMV: x = 0.026% of traffic, −0.07% (RECORDED Type-1). Unfused attn GEMV: +1.315% tps, below 3% gate, +11.1% dispatches (MEASURED). Qwen14 predec-then-GEMV: −29.94% tps, +42.75% wall, +3.041 GB resident (MEASURED). | `QWEN38_AT_CEILING_RESOLVED.json` `measured`. `QWEN38_BANDWIDTH_BOUND.json` `derivation.achieved_gb_s`. `RUNG_QWEN38_MEASURED.json` `rung`. `dead_levers.md:32,35`. `TG32_DEEPSEEK_V2_LITE_UNFUSED_ATTN_GEMV_REJECTED_2026_08_02.json:86-87`. `TG32_QWEN14_GRAVITY_Q4_PREDEC_RESULT_2026_08_02.json` `deltas` / `decision`. | Never as a decode ceiling argument against 98.7%. Reopen kernel work only if the genome is no longer unique-once GEMV (new launch geometry, or a representation-specific kernel whose measured GB/s on this box is shown against a same-shape unique-once control). Do not reopen A5/A6 uint4/TG sweeps. Do not reopen f16-x into predec GEMV. Do not reopen unfused attn GEMV without a new resident attention grammar that also restores unrelated-prompt parity. Do not reopen predec-table expansion without a different scale representation or graph change. |
| 3 | Speculative decode as a Metal tps win (as-is) | KILLS for the shipped ExactShared path and for EAGLE-3 trained heads. External llama.cpp MTP-on-Metal is a known regression class, not a Hawking generate receipt. | ExactShared: RECORDED 0.11 vs 18 tps, serial verify. EAGLE-3: RECORDED τ=0.877 ≪ 2.5; on-device 0.40× / 0.30× / 0.21× vs base; free n-gram beats trained. Code comment: low acceptance makes ExactShared slower than normal decode. llama.cpp MTP: issue score 18, title only, no Hawking number. | `dead_levers.md:13,39`. `crates/hawking-speculate/src/shared.rs:3-7`. `tools/bench/workloads/generated/llama_cpp_spec_decode_regression_mtp_speculative_decoding_degrades_throughput_on_metal.json` `source_issue`. Gate: `tools/bench/workloads/spec_decode_gate.json`. | ExactShared: batched verify **or** a 10–20 ms draft. EAGLE-3: oracle τ ≥ 2.5 first, then on-device. MTP: a Hawking complete-token A/B on this box with 0 fallbacks; do not import the GitHub title as a measured number. Spec remains allowed only if accepted-tokens/sec rises and P95 does not regress (`spec_decode_gate.json`). |
| 4 | Sub-bit results scored on Gaussian / synthetic X | KILLS as a promotion or ranking path. Those “negatives” were proxy artifacts. | GLM: six families first negative under Gaussian. On real teacher-capsule X the ranking inverted; null moved 0.126 → 0.651; activation-aware then 0.755 cosine at 0.167 BPW on 12/12 experts. Top 16 activation directions carried 88.9% of variance. Dead-levers Colab row (older, Gaussian-era): four families 0.116–0.157 cos @ 0.75 BPW, none beat null 0.898 — treat as the artifact this later receipt reversed. | NS-009 in `NEGATIVE_SCIENCE_REGISTER.json:278-297`. `HAWKING_HEAVY_CONTINUATION_STATUS.json` `rebuild_glm.step_2_pilot` lines 25–26. `G016_BPW_FEASIBILITY.json` `blocking_work[1]`. `dead_levers.md:75` (superseded ranking, keep as the artifact). `FS_PER_WEIGHT_LAW.json` `precedent_for_sub_bit`. | Never as a promotion or ranking path. Synthetic X is a kernel numeric fixture only. Reopen a family only on real teacher-forced (or captured BF16) X from the named source, with a stated null, and a generation gate. |
| 5 | Activation-aware vs synthetic evaluation trap | KILLS. Same mechanism as #4, first-class graveyard class. | Input distribution decides which family wins. Qwen3.8 descent and reconstruction-is-free both required `not_synthetic=true` / “explicitly not synthetic”. Attention-density probe: `used_synthetic_or_gaussian=false`. | Graveyard class `synthetic_activation_mismatch`: `ASCENSION_GRAVEYARD_PLAN.md` §4, `lab/operators/ascension_graveyard.py` `FailureClass.SYNTHETIC_ACTIVATION_MISMATCH`. `QWEN38_BPW_DESCENT.json` `activation.not_synthetic`. `QWEN38_RECONSTRUCTION_IS_FREE.json:6`. `QWEN_ATTENTION_DENSITY_VERDICT.json` `activation_honesty`. | Real teacher-forced / captured-BF16 X. Down_proj on post-SwiGLU X, not layer hidden (NS-012). |
| 6 | Compression scored on the wrong baseline ranks the wrong trajectory | KILLS. | MEASURED (recorded as prior-campaign fact): X captured from a 0.7966 gibberish baseline; every score ranked the wrong trajectory. Same-class: mixed-floor / Q4-runtime “0.135% efficiency” (NS-003); 403 ms Q80 “baseline” that was a dirty single-run (NS-035); stale-baseline SIMD 2×/8–45× that became 2.5× slower after reconstruction (91.28 vs 36.53 ms). | NS-015 `NEGATIVE_SCIENCE_REGISTER.json:416-430`. `_Q80_DENSITY_COMMON.md:75-77`. NS-003, NS-035. `STALE_BASELINE_PATTERN.json` case `gk_*_simd`. `Q80_GK_SIMD_NEGATIVE.json:4-7`. | Fit only against the named source (Qwen3.8 BF16, Q80 BF16, DSV4F official mixed). A lane result is valid only against the HEAD it was measured on; rebase + remeasure if a concurrent lane invalidated the baseline. |
| 7 | HGRAVB01 / HGRAVR02 / HGRAVS01 (and the Q80 mixed gate/up/down bundle) on Qwen3.8 attention | KILLS for Q4-equivalent quality. | Attention cannot be cheaply compressed below uniform-Q4 at the 0.99 bar. Binary max cos 0.946 (typical 0.75–0.88). Rice residual max 0.958 (typical 0.82–0.91). SVD r=512 typical 0.83–0.89; 0 clears of 0.99 at ranks that beat Q4 BPW. Q3 on attention: out_proj 0.953 / 0.968, in_proj 0.975–0.980. | `QWEN_ATTENTION_DENSITY_VERDICT.json:5` and `codec_applicability`. `QWEN38_DENSITY_ROOT_CAUSE.json`. | A **new** attention codec family, scored on real BF16 X, clearing mean-row output cosine ≥ 0.990 vs BF16 **and** a multi-prompt generate identity gate. Do not transfer the expert MLP bundle to Q/K/V/O or DeltaNet. |
| 8 | Qwen3.8 mixed-sub15 (~1.29 BPW) and mixed-2p0 (2.0856 BPW) as coherent artifacts | KILLS those two recipes. Floor bracketed, not located. | mixed-sub15: INCOHERENT, 0 fallbacks, two-token cycle `220/264` (“ a”). Packer PROJECTED 79.44 TPS / 12.588 ms — not measured. mixed-2p0: INCOHERENT native, `[198]×15` then `[8]`, 0 fallbacks, 0 dense-W. Q4 4.2527 oracle on the same prompt emits `<think>`. Floor with **current codecs** lies between 2.0856 and 4.2527 BPW. | `QWEN38_SUB15_INCOHERENT.json:12-35`. `QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-15`. | A different recipe (especially a new attention codec; MLP is already 0.848 BPW). Re-test generate natively (no mixed→float→Q4). The interval (2.0856, 4.2527) is untested — a coherent point there is a new measurement, not a retry of these two packs. |
| 9 | Q30 static ≤1.5 as a template | KILLS as a template. | Coherence FAILED. Bits reachable, capability not. Capture bodies deleted 2026-08-15. | NS-011. `QWEN30_SOURCE_IDENTITY_RECLAIMED.json`. | New codec family **and** a generation gate. Not a copy of the Q30 static pack. |
| 10 | Q80 uniform-Q4 group64 as a target / family member | DEAUTHORISED. | complete_physical_bpw 4.259241. Exceeds 1.5 ceiling. Tensors deleted. Correctness oracle only. | NS-007. Steer S003. `G023_REUSE_MATRIX.json` `q4_status`. | Never as a campaign target. Leave the source. |
| 11 | Q80 mixed-1p5-v1 / mixed-sub655 as Genesis vehicles | SEALED. Weights deleted. Science retained. | Tournament 26.2 / 100 vs Qwen3.8 49.6. Artifact self-declares `coherence_class=DEGRADED`. Storage 0.6462 BPW is 2.518 **active** BPW (NS-001/002). | `Q80_SEALED_LOSER_SCIENCE_RETAINED.json`. NS-001, NS-002, NS-006 (later corrected — see #20). | Odyssey reconstruction only. Do not resurrect the Q80 campaign as a G1 vehicle. Steal the science in §“Steal, do not resurrect”. |
| 12 | DSV4F as a live density/velocity vehicle | SEALED. Weights deleted. | 10.280 GB unique bytes/token ÷ 411.51 GB/s = 24.98 ms floor = 40.03 TPS roof, **below** the 20 ms / 50 TPS gate. `runtime_ready=false`. | `DSV4F_SEALED_SCIENCE_WEIGHTS_DELETED.json` `why_dsv4f_left_the_active_fleet`. `DSV4F_GATE_BELOW_DRAM_FLOOR.json`. | A reseal to ≤7.042 GB/token (ESTIMATED 17.11 ms / 58.44 TPS) plus a local execution adapter. Human-authorised. Not a G1 Qwen3.8 action. |
| 13 | Shared cross-expert basis / shared codebook | KILLS on Q80 and on earlier MoE parents. Not settled on DSV4F. | Q80 L10 n=96: gate pairwise cos mean 0.00414279, p95 0.007685; up mean −5.97e-5; top-32 subspace overlap 0.0204. Foundry F0: mean pairwise 1e-4. F1 row-normalized gate 0.00166 vs 0.00168. | `QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json`. NS-010. `tools/foundry/NEGATIVE_TRANSFER_ATLAS.json` `inter_expert_redundancy`, `cross_expert_and_cross_layer_tying`. | Q80: never. Other parent: measure pairwise cosine on **that** parent; reopen only if mean ≳ 0.10 (foundry) / ≳ 0.05 (NS-010 DSV4F note). Do not transfer the 0.004 figure to DSV4F (register attribution correction). |
| 14 | Raw-weight PQ/VQ / uniform sub-bit / ternary-vs-VQ as the route under 1 bit | KILLS those families on frozen weights. | F1 real forward: A1_1p0 at 1.0075 BPW collapsed 6/6 (symKL 7.6–10.9, argmax 0.0). Uniform and treated artifacts collapsed. Ternary dominated by VQ at every matched rate. Entropy coding of Lloyd PQ indices: 0.0–0.7%, not 10–25%. | `NEGATIVE_TRANSFER_ATLAS.json` `raw_weight_pq_vq_at_one_bit`, `uniform_subbit_allocation`, `ternary_factorization`, `entropy_coded_pq_indices`. | Never on raw frozen weights. A method that **changes the source** (QAT, distill, compressibility training) is a different lever. |
| 15 | Q3_K / QTIP Metal / learned codebook / low-rank residual as dense-tps | AXIS CLOSED for tps. | Q3_K: 22–24% peak, slower µs than Q4 despite fewer bytes (RECORDED Type-1). QTIP Metal: more ALU than dead Q3 + serial state (RECORDED Type-1 proxy). Learned codebook: random LUT gather punished on Apple GPU. Low-rank residual: SVD energy low; ASVD also dead; doctor atlas LoRA plateau, rank-64 SVD 0.114→0.104 at +4.6 bpw. | `dead_levers.md:21-22,30-31,77`. `DOCTOR_NEGATIVE_TRANSFER_ATLAS.json` `low_rank_residual`. | Footprint-only for Q3_K. QTIP: lane-independent layout + ≥55% peak oracle. Low-rank: residual **measured** low-rank on this parent, or a model trained low-rank. Learned: lookup-free (QTIP-class) only. |
| 16 | Organ cosine ~0.86–0.90 as a capability certificate | KILLS as a GO. | Constant-mean null 0.898. Q80 0.8604 bar sits **below** that null. mixed-1p5 then generated coherent text with down_proj holdout cosine 0.7684 — so the bar is also not a hard cliff (NS-016). 12-token single-prompt is a smoke test, not lm_head certification. | NS-013, NS-016. `_Q80_DENSITY_COMMON.md:58-64`. `Q80_LM_HEAD_NEGATIVE.json` `A_CAUTION_ABOUT_MY_OWN_GATE`. | Beat a stated null **and** pass generation on more than one prompt. lm_head / embed / sampling: multi-prompt id-identity, not 12 tokens. |
| 17 | Storage / complete_physical BPW as the bytes decode moves | CATEGORY_ERROR. | Q80 mixed-sub655: storage 0.6462 vs active 2.518 (~4×). mixed-1p5: 1.4444 vs 4.98. Batch=1 reads 10 of 512 experts. Qwen3.8 is dense — storage ≈ active except the embed table (one row gathered). | NS-002. `G013_FS_EFFICIENCY_CLOSURE_V2.json`. `QWEN38_BANDWIDTH_BOUND.json` `derivation.embed_note`. | Never as a substitute for active BPW. On Qwen3.8, exclude the embed table from the active budget; include lm_head (greedy reads it). |
| 18 | 535–647 GB/s reuse band as the decode ceiling | CATEGORY_ERROR. | Reuse 64 MiB × 4096: 535.9–637.5 GB/s, flagged `not_the_decode_ceiling=true`. Unique-once 1024 MiB: 301.6. Unique-once 512 MiB: 411.51. Token-shape 98-CB 10-of-512: 319.7–363.2. Published peak 819 is not a measured decode bound. | NS-004. `QWEN38_AT_CEILING_RESOLVED.json` `the_distinction_that_actually_matters`. `G014_FS_OCCUPANCY_CONTROL.json` (cited; lives on `89f40c76e`). | Never as a batch=1 decode ceiling. Valid only for a working set that actually stays hot. |
| 19 | low-BPW → materialize W / expand to Q4/float → generic GEMV | KILLS unless complete-token measurement says otherwise. | Qwen14 predec tables: +3.041 GB resident, −29.94% tps (MEASURED). NS-018: caching decoded W restores the footprint the representation exists to remove (48×512 decoded = 288 GiB vs packed 11.05 GiB). NS-019: Q80 down_proj already does `y=L@(R@x)`; re-opening reconstruct-W does not move the token. mixed-2p0 earlier incoherence went through mixed→float→Q4, which the native reader later removed so the collapse could be attributed to the artifact. | `TG32_QWEN14_GRAVITY_Q4_PREDEC_RESULT_2026_08_02.json` `decision.binding_finding`. NS-018, NS-019. `QWEN38_COHERENCE_FLOOR_BRACKETED.json:5`. `QWEN38_SUB15_INCOHERENT.json` `speed_caveat_carried`. | A representation-specific kernel that consumes codes in-register, **or** a complete-token measurement proving expansion is still a net win. Isolated organ microbench is not that measurement (NS-034). |
| 20 | “Density costs speed” / 5.9× rice reconstruction penalty | SUPERSEDED. Was a kernel-geometry artifact, then wrongly transferred to Qwen3.8. | Original (NS-006): mixed 2.57 vs Q4 15.2 GB/s = 5.9× slower/byte; token 1171 vs 225 ms. Correction (`Q80_RECONSTRUCTION_WON`): in-register tiles 867.0 → 36.6 ms gpu_matvec (23.7×); isolated Q4 serial on the same shape is ~2.5 GB/s, same as mixed. Qwen3.8 (`QWEN38_RECONSTRUCTION_IS_FREE`): 33 codecs at tpr64 land 15,124–15,541 ns vs f32 control 15,125 ns; 32/33 recon-excess = 0. Same codecs at tg256 ~26,500 ns — launch geometry, not codec. Descent screen that used 5.9× is not final (`QWEN38_BPW_DESCENT_REVIEW.json`). | NS-006 (historical). `Q80_RECONSTRUCTION_WON.json:40-43`. `QWEN38_RECONSTRUCTION_IS_FREE.json:4-19`. `QWEN38_BPW_DESCENT_REVIEW.json` `A_CAVEAT_THAT_MAY_INVALIDATE_THE_SCREEN`. `STALE_BASELINE_PATTERN.json` case `BPW descent screen rice penalty`. | Do not cite 5.9×. Codec choice on Qwen3.8 at tpr64 is quality-constrained, not recon-time-constrained. Reopen a recon-cost claim only on a **named launch geometry** with GPU timestamps, same-vehicle, real X. |
| 21 | lm_head below Q8 (Q80) / Q3 lm_head as a silent policy | KILLS as a silent overlay. | Q80 greedy flips vs BF16: Q4 11–13, Q3 37, Q2/binary ~100, SVD r256 349. Cosine 0.99 is not token identity (Q80 Q3 top-1 0.875 vs Q4 0.9427). lm_head is 14.7% of per-token bytes but ~1.25 ms GPU. | `Q80_LM_HEAD_NEGATIVE.json`. `QWEN_ATTENTION_DENSITY_VERDICT.json` `lm_head_q3_candidate`. | Two-pass Q3 draft → top-32 → gather exact rows → rescore, and only if generated ids equal authority full-eval ids on **more than one** prompt. |
| 22 | CB / encoder / dispatch topology collapse as the token lever | KILLS as the primary lever. | Q80 fuse-to-51 CBs: REGRESSED 516 vs 307 ms, bit-identical. DSV4F encoders 731→43, attention GPU did not move. Expert-wave: fewer drains, +71% token. Resident state: 2573 vs 2525 ms (−1.9%, noise). | NS-020. `HAWKING_EXPERT_WAVE_NEGATIVE.json`. `HAWKING_RESIDENT_STATE_NEGATIVE.json`. | A topology change that also changes occupancy or removes GPU-idle host I/O, scored on **warm paired** complete-token wall, not a cold median. Resident state is scaffolding, not a milestone. |
| 23 | Adding stage wins / isolated organ products and calling the sum a token | KILLS. | dsv-admission-identity cut its metric 2.9× and moved the token by nothing. q80-decode-kernels 192 ms = 480 isolated organs, not a token. | NS-021, NS-034, NS-036. `JOINT_TOKEN_NS.json`. | Compose on one binary. Remeasure the complete token. |
| 24 | 4-layer residual probe extrapolated to 48 layers as a GO | KILLS. | 0.3429 × 1.277^44 = 16211 at L48 from a span [0,4] whose growth ratios are 1.733, 1.229, 0.978; does not beat shuffled-weight null; 395/2048 ranks clamped. | NS-017. `Q80_COHERENCE_LAYER_DRIFT_PROBE.json`. | Full-depth or tiled residual probe that separates from a stated null **and** a generation run. |
| 25 | Underdetermined fit (rows < rank or rows < dim) | KILLS the score, not necessarily the codec. | Q80 25k capture: p10=34, p50=258 rows vs 2048 dim; 24326/24576 gate/up underdetermined. Prior run median 92 rows, every score garbage. DSV4F writer defaults 16 or 64 rows vs dim 4096. | NS-014. | Re-score only when `n_fit ≥ claimed rank` (full-dim: `n_fit ≥ dim`), rank not clamped to `n_fit`. |
| 26 | Shader compile / route-id readback / SHA-per-token as the live wall | KILLS as the primary current wall. | Compile ~24 ms on Q30 startup. DSV4F route-id ~0.1 ms total. SHA/stat/parse on immutable blobs is the repeating pattern (Q80 late-token bind 90–245 ms). | NS-022, NS-023, NS-024. | Compile may appear in a cold-start admission claim. Route-id only if a new graph makes it a tens-of-ms blocking wait. Identity belongs at admission. |
| 27 | Persistent 8 GiB expert arena / previous-token route prefetch | KILLS as complete-token wins. | Arena spreads overlap, no clean speedup, 12/12 bit-identical. Prefetch cannot see the (layer, expert) pairs that actually miss. | NS-028, NS-029. | Whole-catalog admission-time prebind (different mechanism). Corpus-level hot-set prebind (different mechanism). |
| 28 | rice_q1 serial bitstream expand on the per-token path | KILLS as a per-token kernel. | 15.597 ms / organ serial. Bind-time expand + CSR apply is the token path. | NS-031. `q80-decode-kernels.json`. | Never as a per-token kernel. Bind-time expand stays. In-register rice at tpr64 is a different genome (#20). |
| 29 | DRAM-row interleave / Gray/LUT code permutation as the token lever | KILLS. | Q4 and binary LOST. Co-route greedy 170.29 → 164.28 (1.037×), weak. Switching-activity GPU 299000 vs 299333 ns, same speed. | NS-026, NS-027. `dram-row-locality.json`. `codec-switching-activity.json`. | Pack-time colocated triplet blob (named, different). Energy claim needs measured joules. |
| 30 | Gather-vs-sequential as the Q80/Qwen3.8 bandwidth explanation | KILLS. | G014 rank 11: 10-of-512 gather vs sequential = 0 ns. Right axis is reuse vs no-reuse. | NS-032, NS-038. | Never. |
| 31 | 0.59 MiB organ vs 64 MiB DRAM-row probe as a “230× occupancy gap” | CATEGORY_ERROR. | Real gap after occupancy ~2.75×, not 230×. Even MLX ~2.3 GB/s isolated on that shape. Qwen3.8-sized ~50 MiB isolated median 182 GB/s is the right comparator. | NS-033. `matvec-mlx-reference.json`. | Never as a Q80 slogan. |
| 32 | Q80 403 ms / 2.479 tok/s as a legal paired GPU baseline | CATEGORY_ERROR. | Single 11-token CPU-wall decode, no paired reps, no GPU timestamps, 1637 fallbacks. Ledger GPU median 175.04 ms, wall 571.51 ms. | NS-035. | Paired, 0-fallback, GPU-timestamped, named vehicle. |
| 33 | “Q80 is host-bound” or “Q80 is bandwidth-bound” without naming the vehicle | CATEGORY_ERROR. | Q4 vehicle: GPU 125 / 225 ms, 15.2 GB/s, 21–27× off 320–411 control. Mixed vehicle (pre-recon): gpu_matvec 863 / 1171 ms, 2.57 GB/s. Post-recon: gpu_matvec 36.6 / 301 ms. | NS-037. `THREE_MODEL_REGIME_SPLIT.json`. `Q80_RECONSTRUCTION_WON.json`. | Always name the vehicle and the genome. |
| 34 | Sub-100 fs/weight on this box with any existing Q80 pack | UNREACHABLE. | Active BPW 2.518 (sub655) / 4.98 (1p5). Floor 765–985 fs at 320–411 GB/s. Unity 819 still 384 fs. Need active BPW 0.256–0.329. | NS-001. `G013_FS_EFFICIENCY_CLOSURE_V2.json`. | New pack that actually moves attention + lm_head into that active-BPW band, or a unique-once control in the real 98-CB / 10-of-512 shape measured above ~411 GB/s. |
| 35 | Crushing routed experts to move Q80 token time | KILLS as a velocity lever. | Attention 73% of per-token bytes; attention+lm_head 86–88%; routed experts 9%. | NS-005. | Only if routing changes so a much larger expert fraction is actually read per token. |
| 36 | Determined ~20 h DSV4F teacher-X capture | DEAUTHORISED. | Cannot close G007. Daemon exclude list. | NS-008. | Human authorisation, or a cheaper layer-tiled discard-X design (new premise). |
| 37 | Giant JSON capture index | KILLS. | 1.38 GB `capture-result.json` was an iteration wall. | NS-025. | Layer-tile, fit, discard X. |
| 38 | Expert merge / router distill / Kronecker-at-depth / 88-token calibration | KILLS on the parents they were measured on. | Merge: best survivor rel-err 0.885/0.993/0.995. Distill: no student beat masking 0.0784. Kronecker: DEAD L≥1 (top component 0.27% energy); LIVE on L0 (0.0301 vs 0.2252). 88-token calib: 63.6% route-split stable, 26.1% cells never route. | `NEGATIVE_TRANSFER_ATLAS.json`, `DOCTOR_NEGATIVE_TRANSFER_ATLAS.json`. | Merge: best-single-survivor held-out rel-err ≤ 0.5. Distill: large survivor-restricted oracle gap **and** a student that generalizes. Kronecker: non-flat Van Loan spectrum (check L0 separately). Calib: ≥1000 tokens. |
| 39 | Cross-engine / host-dispatch families on M3 UMA (AMX, ANE, ICB, prefetcher, multi-CQ, …) | Type-1 DEAD. | AMX can't eat Q4_K. ANE 4–7× slower. ICB encode 0.27% of GPU. Prefetcher WILLNEED −29%. Multi-CQ: kernels already ≥2× saturated. MTLHeap 8.5% slower than shared. | `dead_levers.md:50-62`. | New GPU generation or a workload that is no longer GPU-saturated decode. |
| 40 | CPU+GPU pipeline / host encode / ICB as tps | Type-1 DEAD. | Greedy already one TCB. CPU encode 0.51% wall. CPU decode 0.06 tps, steals bus. | `dead_levers.md:11,17,19`. | Non-greedy CPU-heavy sample, or encode >1 ms/tok. |

---

## Named cases (expanded)

### Megakernel

Three different objects share the word. Do not collapse them.

1. **MoE megakernel-deeper** (`dead_levers.md:25`). RECORDED Type-1: gate+up+SiLU already fused; remaining hand-off 0.04% wall. Primary timing receipt for the 0.04% figure was not located in this sweep (bible citations in the same file: `v110-path30-findings`, `bible-execution-2026-05-30`). Treat 0.04% as RECORDED, not re-verified.
2. **Qwen-3B full-layer megakernel POC** (`megakernel.rs:1-8`, `megakernel_qwen3b.metal` header, test `megakernel_2layer_parity.rs` `#[ignore = "megakernel POC"]`). Shader stages A..L are TODOs. Dispatcher correctness gate is `x_out == x_in`. Weights are pre-dequantized f16. Hardcoded Qwen-3B shape. This is an abandoned skeleton, not a falsified production kernel.
3. **Fused expert-wave / CB collapse** (measured). `HAWKING_EXPERT_WAVE_NEGATIVE.json`: 1413 vs 828 ms, +71%, bit-correct, flag stays off. Falsifies “minimize synchronization count”. `HAWKING_RESIDENT_STATE_NEGATIVE.json`: 2573 vs 2525 ms, −1.9%, prerequisite not a paying step. NS-020: Q80 516 vs 307 ms. NS-030: redundant-R fusion lost 5–13× because Metal has no grid-wide barrier.

G1 reopen is (3) only, and only with occupancy evidence. (1) needs a new sync primitive. (2) is not a result.

### GEMV axis

Closed for **Qwen3.8 on the current unique-once Q4 GEMV genome**. Evidence chain:

- Active bytes 13.618–13.622 GB/token (embed table excluded). `QWEN38_TOKEN_NS_LEDGER.json` `weight_bytes.active_bytes` = 13618141856. `QWEN38_BANDWIDTH_BOUND.json` `active_bytes_per_token` = 13621829601.
- Achieved 352–406 GB/s depending on whether lm_head is counted. Both ends sit at the top of the 320–411 unique-once control.
- Honest ceiling 411.51 GB/s (`QWEN38_AT_CEILING_RESOLVED.json`). 406.2 / 411.51 = 98.7%.
- Rung: 38.217 ms wall, 26.17 tok/s, roof at current bytes 30.21 tok/s. Rung A (50 TPS) is byte-blocked: max 8.230 GB/token at 100% of 411.51 with zero host tax (`RUNG_QWEN38_MEASURED.json` `reachable_at_current_bytes`).
- Consequence written in that receipt: Qwen3.8’s lever is fewer bytes, not kernels. The hoped-for ~2× kernel headroom does not exist on this genome.

Separately closed, older, still in force on this box:

- Decode micro-opt A5/A6: GEMV already at memory-model optimum (`dead_levers.md:35`).
- f16-x into predec GEMV: 0.026% of traffic (`dead_levers.md:32`).
- Q4_K MMA rows≤cols: square/wide lose; tall +22–24% (`dead_levers.md:33`).
- Unfused attn GEMV +1.315% < 3% gate.
- Predec-then-GEMV on Qwen14: −29.94% tps. This is the expand-then-GEMV binding in numbers.

Q80 is the opposite genome: 15.2 GB/s vs 320–411, a 21–27× kernel-shaped gap (`G001_KERNEL_GAP.json`). That gap is Q80’s, not Qwen3.8’s. Do not import it.

### Spec-decode

Hawking-owned numbers in this tree are thin.

- ExactShared 0.11 vs 18 tps: only `dead_levers.md:39`. No primary generate receipt located. Code in `hawking-speculate/src/shared.rs:3-7` documents the mechanism (shared-expert draft, serial prefix verify, low acceptance ⇒ slower than greedy) but does not reprint the 0.11 figure.
- EAGLE-3 τ=0.877 / 0.40×/0.30×/0.21×: only `dead_levers.md:13`. Primary oracle receipt not located (`v110-path30-findings` cited, not present as a file in this sweep).
- llama.cpp MTP-on-Metal: workload JSON cites `ggml-org/llama.cpp#23752`. Not a Hawking measurement.

Cheapest experiment that would turn RECORDED into MEASURED: one paired 0-fallback generate on this box, ExactShared on vs off, GPU timestamps, accepted-tokens/sec and P95. Until then do not spend a G1 lane re-deriving EAGLE-3.

### Sub-bit Gaussian-proxy artifact

The project’s own correction, not an external opinion.

`HAWKING_HEAVY_CONTINUATION_STATUS.json:25-26`:

> Six families first measured negative sub-bit under GAUSSIAN proxy activations. Refitted against real captured teacher-capsule activations, the ranking inverts and the null moves from 0.126 to 0.651.

Activation-aware then 0.755 cosine @ 0.167 BPW on 12/12 experts. Raw-weight low-rank: 0/12 beat the null at 0.667 BPW. Structural reason: top 16 activation directions carry 88.9% of variance; weights live in 6144 dimensions the model never visits.

`dead_levers.md:75` still lists the Gaussian-era “four families 0.116–0.157 cos @ 0.75 BPW, none beat null 0.898”. That row is the artifact NS-009 warns against rediscovering. The later reversal is the live law.

G016 wrote the transfer into the Qwen3.8 packer brief: “Synthetic activations are known-invalid here: every prior sub-bit negative was a gaussian-proxy artifact.”

### Wrong-baseline ranking

NS-015 / `_Q80_DENSITY_COMMON.md:75-76`: a prior campaign captured X from a **0.7966 gibberish** baseline and every score ranked the wrong trajectory. Teacher must be the BF16 (or official mixed-precision) source.

Same trap, different clothes:

- NS-003: mixed-artifact floor over Q4 runtime = 0.135% “efficiency”. Superseded the same day.
- NS-035: 403 ms Q80 baseline, dirty single-run, 1637 fallbacks.
- `STALE_BASELINE_PATTERN.json`: SIMD “2× on binary, 8–45× on Q8” was true against the serial extract and false against the shipping tiles (91.28 vs 36.53 ms, 2.5× slower).
- `QWEN38_COHERENCE_FLOOR_BRACKETED.json:5`: mixed→float→Q4 made earlier incoherence un-attributable. Native reader required.

### Sealed codec families (do not resurrect as G1 vehicles)

| family / artifact | status | why | reopen |
|---|---|---|---|
| Q80 uniform-Q4 group64 as target | DEAUTHORISED S003 | 4.259 BPW > 1.5 | correctness oracle only |
| Q80 mixed-1p5-v1, mixed-1p5-ne4, mixed-sub655 | SEALED, weights deleted | tournament loser; mixed-1p5 `DEGRADED`; science retained | Odyssey rebuild from catalog + fit tables |
| Q30 static ≤1.5 | FAILED coherence | bits ≠ capability | new family + generate gate |
| DSV4F streamed 43-layer | SEALED, chunks deleted | 24.98 ms DRAM floor > 20 ms gate; no runtime adapter | reseal ≤7.042 GB/token + adapter |
| HGRAVB01 / R02 / S01 on attention | quality fail | miss 0.99 by a wide margin | new attention family |
| Qwen3.8 mixed-sub15-v1 (~1.29) | INCOHERENT | degenerate 220/264 cycle, 0 fallbacks | different recipe, native generate |
| Qwen3.8 mixed-2p0-v1 (2.0856) | INCOHERENT native | 15 newlines + `)` | different recipe, especially attention |
| raw-weight PQ/VQ ~1 bit | DEAD family | 6/6 collapse at 1.0075 BPW | change the source, do not recode frozen W |
| Q3_K as tps | Type-1 | slower than Q4 | footprint only |
| QTIP Metal trellis | Type-1 proxy | ALU + serial state | lane-independent layout + ≥55% peak |
| learned LUT codebook | dead | gather punished | lookup-free only |
| shared cross-expert basis | refuted | cos ~0.004 | other parent, measured cos ≳ 0.05–0.10 |
| rice_q1 **per-token serial** | refuted | 15.6 ms/organ | bind-time expand; in-register rice is #20 |
| Q80 lm_head HGRAVB/R/S or silent Q8→Q4 | forbidden | token flips from Q4 down | two-pass rescore, multi-prompt ids |
| GLM-5.2 Generation A PQ R0 | deleted | Gaussian-era negative control | N/A |
| HQ80BR1 velocity thesis | not supported by mapped evidence | 3.09× expert cut = 0.23% of a 341.9 ms token at the then-quoted 5.55 GB/s | do not build expecting speed (`docs/HQ80BR1_KERNEL_DESIGN_SUPERSEDED.md` §0). Note: that 700 GB/s “ceiling” is a blit roofline, not the unique-once decode ceiling (NS-004). |

`workspace/docs/history/HAWKING_PACKS_RETIREMENT.json` is a repo-absorption receipt, not a codec kill.

`workspace/campaign/governance/odyssey/state/graveyard/GRAVEYARD.json` is a schema stub (`contents` / `revival` only). `lab/operators/ascension_graveyard.py` is a scaffold; remaining work item 1 in the plan is “seed burials from existing ledgers”. This file is that seeding for G1.

---

## Steal, do not resurrect (Q80 / DSV4F science)

Contract: do not resurrect those campaigns as vehicles. The transferable laws:

- Active BPW ≠ storage BPW on MoE. On dense Qwen3.8 they nearly coincide except embed.
- Attention, not experts, is the mass that sets the token on both Q80 (73% of per-token bytes) and Qwen3.8 (74% of the 2.0856 artifact).
- Same-vehicle efficiency only.
- Unique-once, not reuse, is the decode ceiling.
- Reconstruction cost is launch-geometry-conditioned. In-register at tpr64, it is free on Qwen3.8 (MEASURED, 33 codecs).
- Organ cosine is a screen. Generation is the gate. Null must be stated.
- Down_proj inverts family ranking and needs post-SwiGLU X.
- `rank = min(budget, n_fit_rows)` silently starves the codec.
- A component microbench is not a token. A projected TPS from packed bytes is not a token (`QWEN38_SUB15_INCOHERENT.json` 79.44 TPS was PROJECTED).
- Identity of an immutable blob is an admission cost.
- Q80 `expert_bind` 111.7 ms/token (57%) was measured and never removed — only relevant if Q80 is rebuilt.

---

## Superceded numbers — do not cite as law

From `NEGATIVE_SCIENCE_REGISTER.json` `superseded_do_not_cite_as_law` plus later same-day corrections:

| number | why dead |
|---|---|
| 0.135% Q80 efficiency | mixed floor / Q4 runtime (NS-003) |
| 560–647 GB/s decode ceiling | cache-resident reuse band (NS-004) |
| sub-100 fs needs BPW < 0.448–0.518 | used the reuse band and storage BPW (NS-001) |
| Q80 mixed floor 757 µs / 1321 tok/s at 1.392 BPW | storage-ish BPW × active-weight count |
| 5.9× rice/binary/low-rank recon penalty | serial extract, then transferred ( #20 ) |
| gk_*_simd 2× / 8–45× unused headroom | subsumed; 2.5× slower than tiles |
| 230× occupancy gap | 0.59 MiB organ vs 64 MiB probe (NS-033) |
| 403 ms / 2.479 tok/s Q80 baseline | dirty single-run (NS-035) |
| 16211× residual at L48 | 4-layer extrapolation (NS-017) |
| DSV4F experts mutually orthogonal at cos 0.004 | that 0.004 is Q80 L10, 96 of 512 |

---

## G1-conditioned reading

G1 target (pursue honestly, not a fabrication pass): capability preserved, complete effective BPW < 1.5, TOKEN_NS ≤ 10,000,000, TPS ≥ 100.

What the ledger says about those numbers, without this lane measuring anything:

- At current bytes, the honest roof is ~30 tok/s (`RUNG_QWEN38_MEASURED.json` `roof_tok_s` = 30.209). Rung B (100 TPS / 10 ms) at 411.51 GB/s and zero host tax needs ≤ 4.115 GB/token (`byte_budgets.B_max_bytes` = 4115135858.96). That is PROJECTED from the unique-once ceiling, not a measured 100 TPS.
- Current complete BPW 4.2527. 4.115 GB / 13.618 GB × 4.2527 ≈ 1.285 BPW PROJECTED to sit on the 10 ms byte budget, **if** the genome stays unique-once GEMV at 411 GB/s and host tax is zero. Both ifs are genome claims.
- Current codecs are INCOHERENT at 2.0856 and 1.29. The coherence floor is above 2.0856. Hitting < 1.5 **and** coherent therefore requires a **new attention codec family**, not another HGRAVB/R/S pack. MLP is already 0.848 BPW.
- Reconstruction time is not the constraint at tpr64. Quality is.
- Kernel micro-opt on the current Q4 GEMV is closed. A representation-specific kernel is a new genome and is the binding-compliant path; it must still be scored as a complete token against a unique-once control, not against 819 or 560–647.

Cheapest experiment this ledger cannot replace: a native generate on one new attention recipe in the open interval (2.0856, 4.2527), 0 fallbacks, multi-prompt, GPU-timestamped. That locates the floor. It is not a retry of mixed-2p0 or mixed-sub15.

---

## Evidence excerpts

### NS-009 Gaussian proxy (HEAD receipt)

`receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json:278-297`

```
"id": "NS-009",
"mechanism": "Evaluate or fit compression on Gaussian / synthetic proxy activations",
"what_was_measured": {
  "glm52": "Six families first measured negative sub-bit under Gaussian proxy. Refitted on real teacher-capsule X, ranking inverted and the null moved from 0.126 to 0.651. Activation-aware then hit 0.755 cosine at 0.167 BPW on 12/12 experts."
},
"why_it_failed": "The input distribution decides which family wins. Gaussian X is not a detail. Every prior sub-bit negative in this project traced to that proxy.",
"retry_when": "never as a promotion or ranking path. Synthetic X is allowed only as a kernel numeric fixture, never as a codec-selection or sub-bit evidence input."
```

### NS-015 wrong baseline

`receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json:416-430`

```
"id": "NS-015",
"mechanism": "Calibrate or capture X from a degraded / quantized / gibberish baseline",
"what_was_measured": "A prior campaign captured X from a 0.7966 gibberish baseline and every score ranked the wrong trajectory.",
"retry_when": "never. Fit against the source named in the lane brief ..."
```

### Qwen3.8 coherence floor (native, 0 fallbacks)

`receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-15`

```
"COHERENCE_FLOOR_BRACKETED": {
  "4.2527_BPW_q4_oracle": "COHERENT",
  "2.0856_BPW_mixed-2p0-v1": "INCOHERENT (native, verified twice - lane and controller)",
  "1.2910_BPW_mixed-sub15-v1": "INCOHERENT",
  "conclusion": "Qwen3.8's coherence floor with current codecs lies between 2.0856 and 4.2527 BPW."
}
"honest_note": "This does not prove sub-2.0 BPW Qwen3.8 is impossible. It proves THIS recipe fails there. The MLP compresses to 0.848 BPW fine; it is attention at 4.250 and 74% of the artifact that dominates, and no attention codec below Q4 exists yet."
```

### mixed-sub15 generate (MEASURED, 0 fallbacks)

`receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json:12-35`

```
"RESULT": "INCOHERENT. FAILS criterion 1 of the G006 coherence bar - degenerate cycle."
"generated": "  a    a  a  a  a  a  a"
"note": "a two-token cycle between 220 (space) and 264 (' a')"
"projection_the_packer_reported": { "projected_tps": 79.44, "implied_bpw": 1.291 }
```

### Reconstruction is free at tpr64 (MEASURED, real X)

`receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json:4-17`

```
"claim": "At the production tpr64 launch geometry, reconstruction costs NOTHING on Qwen3.8."
"activation": "REAL captured BF16 post-norm hidden ... explicitly not synthetic"
"codecs_at_tpr64_ns": "15,124-15,541 for q4/q3/q2/binary/ternary/additive_q2q2/hadamard/rice - i.e. the SAME as uncompressed f32"
"same_codecs_at_tg256_ns": "~26,500 - the penalty is LAUNCH GEOMETRY, not the codec"
```

### GEMV axis closed (MEASURED)

`receipts/ascent-2026-08-16/QWEN38_AT_CEILING_RESOLVED.json` `measured`:

```
"qwen38_achieved_gb_s": 406.2,
"honest_decode_ceiling_gb_s": 411.51,
"pct_of_ceiling": 98.7,
"reuse_band_is_not_the_ceiling": true
```

`receipts/ascent-2026-08-16/RUNG_QWEN38_MEASURED.json:7-9,61`

```
"bytes_per_token": 13622000000.0,
"ns_per_token": 38216792.0,
"gpu_ns_per_token": 36987000.0,
"rung.tps": 26.166508167404526
```

### Attention families do not transfer

`receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json:5`

```
"claim": "Attention GEMVs cannot be cheaply compressed below uniform-Q4 at Q4-equivalent output quality. Existing Gravity expert families (HGRAVB01 / HGRAVR02 / HGRAVS01) do not transfer."
```

### 5.9× correction

`receipts/ascent-2026-08-16/Q80_RECONSTRUCTION_WON.json:40-43`

```
"I recorded 'density is costing speed - mixed is 5.9x slower per byte than Q4 because binary/rice/low-rank reconstruction is expensive'. The codecs were never the cause. Q4's 15.2 GB/s came from a DIFFERENT KERNEL (simdgroup); isolated Q4 serial on the same shape is ~2.5 GB/s, the same as mixed. The cause was a kernel choice, not codec cost."
```

### Megakernel POC (not a measured kill)

`crates/hawking-core/src/kernels/megakernel.rs:1-8`

```
//! Megakernel POC dispatcher (2026-05-25, build/megakernel — day 3+).
//! The shader body is still a pass-through (stages A..L TODO), so this
//! dispatcher's correctness gate is the pass-through invariant `x_out == x_in`,
//! NOT a full per-layer parity vs CPU.
```

### ExactShared (code + condensed kill)

`crates/hawking-speculate/src/shared.rs:3-7`

```
//! ExactShared is intentionally conservative: draft tokens come from the
//! shared-expert-only path, verifier tokens come from the full model, and the
//! accepted prefix is the longest greedy prefix where both agree. Low
//! acceptance makes this slower than normal decode, so callers should keep it
//! as an experimental path rather than a headline performance mode.
```

`workspace/docs/guides/dead_levers.md:13,25,35,39,77`

```
| EAGLE-3 trained head | NO-GO | τ=0.877≪2.5; on-device 0.40×/0.30×/0.21× vs base; free n-gram beats trained | Oracle τ≥2.5 first |
| MoE megakernel deeper | dead | gate+up+SiLU fused; remaining hand-off 0.04% wall | New cross-TG sync primitive |
| Decode micro-opt A5/A6 | dead Type-1 | uint4 unpack / TG sweeps NO-CHANGE; GEMV at memory-model optimum | Not micro-opt; bytes/spec/stateful |
| Spec ExactShared as-is | dead | 0.11 vs 18 tps; serial verify | Batched verify or 10–20 ms draft |
**Axis closed for tps:** Q3_K, low-rank residual, learned codebook, QTIP Metal decode.
```

### Expert-wave measured collapse

`workspace/campaign/evidence/systems/hawking/HAWKING_EXPERT_WAVE_NEGATIVE.json:4-8`

```
"verdict": "NEGATIVE. Correct, gated, and slower. NOT promoted; flag remains off by default."
"measured": {"expert_wave_plus_bf16_ms": 1413, "bf16_only_ms": 828, "ratio": 0.59, "regression_pct": 71}
"what_this_falsifies": "The assumption that synchronization count is the thing to minimize."
```

### Expand-then-GEMV (Qwen14 predec)

`workspace/campaign/evidence/runtime/tg/TG32_QWEN14_GRAVITY_Q4_PREDEC_RESULT_2026_08_02.json` `deltas` / `decision`:

```
"tps_percent": -29.94311199477101
"decode_wall_percent": 42.74822032284287
"resident_bytes_percent": 33.86469157529478
"binding_finding": "Predecoded Q4_K tables were resident and physically used, but added 3,041,445,760 bytes of device state and slowed complete-token wall by 42.748 percent"
"reopen_condition": "Only a different scale representation or graph-level change may reopen this floor."
```

### GLM reversal (primary)

`workspace/campaign/evidence/systems/hawking/HAWKING_HEAVY_CONTINUATION_STATUS.json:25-26`

```
"the_reversal": "Six families first measured negative sub-bit under GAUSSIAN proxy activations. Refitted against real captured teacher-capsule activations, the ranking inverts and the null moves from 0.126 to 0.651."
```

### Generation is the gate

`workspace/ops/ascent-lanes/_Q80_DENSITY_COMMON.md:58-64,75-77`

```
- raw activation cosine is a deceptive metric here (measured null baseline 0.898 —
  i.e. cosine 0.898 can mean NOTHING).
**Generation is the gate.** Nothing else counts.
- Never calibrate on a degraded baseline. A prior campaign captured X from a 0.7966
  gibberish baseline and every score ranked the wrong trajectory.
- Never evaluate compression on synthetic/Gaussian activations — every sub-bit
  negative from that era was an artifact of the proxy.
```

---

## What this sweep could not settle

- Primary receipts behind `dead_levers.md` condensed numbers (EAGLE-3 τ=0.877, ExactShared 0.11 vs 18, megakernel-deeper 0.04% wall, Q3_K 22–24% peak, f16-x 0.026%). Those files are cited as `[[v110-path30-findings]]` etc. and were not present as paths in `git ls-tree HEAD`. Cheapest close: locate those bible notes or re-measure the one G1 actually wants.
- DSV4F pairwise expert cosine. Register says it was never measured; do not invent it.
- Coherence in (2.0856, 4.2527) BPW. Bracketed, not located.
- Odyssey / Ascension graveyards are empty scaffolds. Operational truth is the NS register + this file.

---

## Completion report

```
STATUS
IMPLEMENT_READY

CLAIMS
1. Qwen3.8 GEMV/kernel axis is closed on the current unique-once Q4 genome at 98.7% of 411.51 GB/s. Evidence: receipts/ascent-2026-08-16/QWEN38_AT_CEILING_RESOLVED.json measured; QWEN38_BANDWIDTH_BOUND.json derivation.achieved_gb_s; RUNG_QWEN38_MEASURED.json rung.
2. Current-codec Qwen3.8 coherence floor lies between 2.0856 and 4.2527 BPW; mixed-sub15 and mixed-2p0 are incoherent with 0 fallbacks. Evidence: QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-15; QWEN38_SUB15_INCOHERENT.json:12-35.
3. HGRAVB01/R02/S01 do not transfer to attention. Evidence: QWEN_ATTENTION_DENSITY_VERDICT.json:5; QWEN38_DENSITY_ROOT_CAUSE.json.
4. Gaussian/synthetic X inverted codec ranking (null 0.126→0.651; later 0.755 @ 0.167 BPW on real X). Evidence: NEGATIVE_SCIENCE_REGISTER.json NS-009; HAWKING_HEAVY_CONTINUATION_STATUS.json:25-26.
5. Wrong baseline (0.7966 gibberish X) ranks the wrong trajectory. Evidence: NS-015; _Q80_DENSITY_COMMON.md:75-76.
6. 5.9× reconstruction penalty is superseded; at tpr64 recon excess is 0 on 32/33 Qwen3.8 codecs. Evidence: Q80_RECONSTRUCTION_WON.json:40-43; QWEN38_RECONSTRUCTION_IS_FREE.json:4-17.
7. Megakernel-as-tps is dead on three grounds (0.04% RECORDED remainder; POC pass-through; MEASURED +71% expert-wave). Evidence: dead_levers.md:25; megakernel.rs:1-8; HAWKING_EXPERT_WAVE_NEGATIVE.json:4-8.
8. Spec ExactShared / EAGLE-3 are RECORDED kills; primary receipts not located. Evidence: dead_levers.md:13,39; hawking-speculate/src/shared.rs:3-7.
9. Expand-then-GEMV lost 29.94% tps / +3.041 GB on Qwen14 predec. Evidence: TG32_QWEN14_GRAVITY_Q4_PREDEC_RESULT_2026_08_02.json deltas/decision.
10. Q80 mixed, Q80 uniform-Q4-as-target, Q30 ≤1.5, DSV4F streamed, raw-weight PQ/VQ ~1 bit, shared-expert basis are sealed or refuted as vehicles. Evidence: Q80_SEALED_LOSER_SCIENCE_RETAINED.json; NS-007/011; DSV4F_SEALED_SCIENCE_WEIGHTS_DELETED.json; NEGATIVE_TRANSFER_ATLAS.json raw_weight_pq_vq_at_one_bit; QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json.

EVIDENCE
git show HEAD:receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json (38 entries, counts.entries=38)
git show HEAD:workspace/docs/guides/dead_levers.md (nl lines 11-77)
git show HEAD:receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json
git show HEAD:receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json
git show HEAD:receipts/ascent-2026-08-16/QWEN38_AT_CEILING_RESOLVED.json
git show HEAD:receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json
git show HEAD:receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json
git show HEAD:receipts/ascent-2026-08-16/RUNG_QWEN38_MEASURED.json
git show HEAD:workspace/campaign/evidence/systems/hawking/HAWKING_HEAVY_CONTINUATION_STATUS.json
git show HEAD:workspace/campaign/evidence/systems/hawking/HAWKING_EXPERT_WAVE_NEGATIVE.json
Quoted blocks above carry path:line.

CHANGES
created workspace/superwave/g1/g1-arch-negative.md
no other paths touched

TESTS
see final-message TESTS block (test -s, wc -l, git status --porcelain)

RISKS
RECORDED rows (EAGLE-3, ExactShared 0.11, megakernel 0.04%) have no primary receipt in this tree. Citing them as MEASURED would be a lie. G0 26.17 tok/s / 38.217 ms is the receipt, not a remeasure. Coherence floor is a bracket, not a point.

UNRESOLVED
primary bible receipts for condensed dead_levers numbers
DSV4F pairwise cosine
coherence in (2.0856, 4.2527)
empty operational graveyard scaffolds

NEXT
G1: new attention codec family scored on real BF16 X + native generate. Do not retry HGRAVB/R/S on attention, Gaussian X, mixed-sub15, mixed-2p0, expand-then-GEMV, megakernel-deeper, or ExactShared-as-is.
```
