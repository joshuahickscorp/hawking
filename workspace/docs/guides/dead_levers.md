# Dead levers — canonical kill-ledger (do not re-spawn)

**Authority:** sole kill registry for hawking. Merges throughput-bible kills, the 16-solution silicon audit (from deleted untracked `silicon-builds/`), and 2026-06-01 Colab sub-Q4 quality gates. Bible kill protocol points here; bible keeps only Phase-A worked examples. Update when a lever dies.

Before re-spawn: verify evidence is current; **never re-test a Type-1 kill**; resurrect Type-2 only behind its named cheap oracle. Tokens **dead**, **NO-GO**, and **reject** below are deliberate BB BC-ACCEL-012 content.

## Bible / throughput kills (dense)

| Lever | Status | Evidence (condensed) | Resurrection |
|---|---|---|---|
| CPU+GPU pipelining | dead Type-1 | Greedy already one TCB; CPU encode 0.51% wall; broader CPU decode 0.06 tps, steals bus | Non-greedy CPU-heavy sample or encode >1 ms/tok |
| Cross-layer weight delta | dead | Cosine≈0; delta anti-compressible; data-aware also NO-GO | Model trained for cross-layer structure |
| EAGLE-3 trained head | NO-GO | τ=0.877≪2.5; on-device 0.40×/0.30×/0.21× vs base; free n-gram beats trained | Oracle τ≥2.5 first |
| Eagle5 v1 routing mask | dead | MoE balance 0.987–0.995; no concentration | Balance <0.95 on new corpus |
| f16 residual stream | dead | Accum error after 27 layers corrupts logits | bf16 + proven BW win |
| FFN block-256 sparsity | dead Type-1 | Oracle skip 0.2% @99% recall; neurons sparse but scattered; co-act permute NO-GO | Trained-for sparsity only |
| Host per-dispatch overhead | dead Type-1 | Concurrent QKV +1.68%; PSO free; CPU 0.51%; gap is GPU-side | Encode ≫1 ms/tok |
| Phase 2.2 trivial-op fusion | dead-for-tps | Rope/add/memcpy fusions sub-noise or break bit-id; llama fusion already present | silu_mul-into-down only behind +3% A/B |
| ICB | dead | CPU encode 0.22 ms = 0.51% wall | Encode >1 ms/tok |
| KV working-set eviction | NO-GO Type-1 | 99% mass needs 78–92% positions at short code ctx | Longer/non-code captures + frac99<0.25 |
| Low-rank residual codec | NO-GO Type-1 | SVD energy low; data-aware ASVD also dead | Model trained low-rank; else QTIP path |
| Learned codebook | dead | Random LUT gather punished on Apple GPU | Lookup-free learned (QTIP-class) only |
| W4A8 default decode | held | 20% bit-id; 1.115× <1.20 ship; sub-additive vs predec | Logit quality metric + AWQ-from-f16 |
| MLA Phase 4 simdgroup attn | dead | −1.7..−2.5% regression; attn only 2.4% wall | New intrinsic + long-ctx share |
| MoE megakernel deeper | dead | gate+up+SiLU fused; remaining hand-off 0.04% wall | New cross-TG sync primitive |
| MoE serial route dispatch | dead | Serial 50 ms > parallel 44 ms | Zero-cost encoder switch |
| Phase Y sumy Q4_K v3 | dead | −14% from register pressure | Leaner sumy geometry if Q4_K BW-bound again |
| Predec 4-row ILP default | parked | `_2r` wins; 3r/4r/8r flat/regress; `PAIR_2R_INLINE` shipped +9.6% | xctrace occupancy-bound only |
| Multiseq v4r high-B | NO-GO Type-1 | B=8 −14.7% vs v3w | Different high-B kernel, not v4r re-route |
| Q3_K sub-Q4 byte-cut | NO-GO Type-1 | 22–24% peak; slower µs than Q4 despite fewer bytes | Footprint-only; tps via QTIP reframe |
| QTIP Metal trellis decode | NO-GO Type-1 proxy | More ALU than dead Q3 + serial state | Lane-independent layout + ≥55% peak oracle |
| f16-x into predec GEMV | dead Type-1 | x=0.026% of traffic; −0.07% noise | Not on predec GEMV |
| Q4_K MMA rows≤cols | dead Type-1 | Tall rows>cols +22–24%; square/wide lose | Multi-SG tile + occupancy oracle |
| Q5_0 simd_shuffle | dead | −3.5%; HW already coalesces | New GPU gen |
| Decode micro-opt A5/A6 | dead Type-1 | uint4 unpack / TG sweeps NO-CHANGE; GEMV at memory-model optimum | Not micro-opt; bytes/spec/stateful |
| A10 access-order repack | dead Type-1 | −16.8%; de-coalesces simdgroup | Do not re-test |
| Q8-KV layer-diff | dead | Uniform routing; no layer signal | Skewed model balance <0.95 |
| Semantic cache | Type-2 parked | +1.48 mean / +0 median over exact prefix on git proxy | Real interleaved session oracle |
| Spec ExactShared as-is | dead | 0.11 vs 18 tps; serial verify | Batched verify or 10–20 ms draft |
| LM head simdmat tps | dead | LM head ~4% not 70% | Larger vocab share |
| Vocab norm-bound screen | NO-GO Type-1 | Certificate needs cos>1; rare high-norm rows | Tighter certificate oracle |
| GPU-sat idle reclaim | NO-GO Type-1 | Prod inter-dispatch idle 0.0 ms single-CB | Llama kernel technique only via Instruments |

## Silicon-architecture audit (16 solutions; transcribed 2026-06-01)

**Scoreboard:** 2 LIVE→SHIPPED (#8 Q4_K simdgroup-MMA shape-gated; #13 zero-copy mmap MTLBuffer). 1 prize-redirect (#16→AWQ). 1 held (#15 int4 KV quality). 1 deferred (#7 GPU top-K). **11 dead Type-1.**

| # | Solution | Verdict | Why |
|--:|---|---|---|
| 1 | Hybrid AMX+GPU | DEAD T1 | GPU beats AMX; AMX can't eat Q4_K |
| 2 | Super-page mempool | DEAD T1×2 | arm64 refuses 2MB; sequential stream TLB-amortized |
| 3 | MTLHeap residency v2 | DEAD T1 | 8.5% slower than shared buffers on UMA |
| 4/9 | ICB / argbuf | DEAD T1 | Host encode 0.27% of GPU budget |
| 5 | Non-GEMM CPU offload | DEAD T1 | 3.2% wall + serial deps |
| 6 | ANE CoreML FFN | DEAD T1 | 4–7× slower; no Q4_K |
| 10 | Weight prefetcher | DEAD T1 | Warm cache; WILLNEED −29% |
| 11 | Q4_K+AMX fused | DEAD T1 | BLAS needs materialize; GFLOPS wall |
| 12 | mlock allocator | DEAD coexist | Zero speedup; pins RAM vs slm |
| 14 | Multi-CQ | DEAD prod | Decode kernels ≥2× saturation |
| 16 naive | Mixed-prec RTN | DEAD T1 | Uniform sensitivity; need AWQ-from-f16 |

**Convergent:** on M3 UMA, live decode levers are GPU kernels + load-path memory; cross-engine and host-dispatch families are dead.

### Not kills
- **#7 top-K sampler** deferred (greedy already GPU).
- **#15 int4 KV** held: per-channel redesign cosine 0.982–0.993 on outliers; per-row real-input collapse; PPL gate remaining.
- **#8/#13** shipped in main.

## Colab sub-Q4 quality (2026-06-01) + MoE path

| Gate | Status | Note |
|---|---|---|
| imatrix mixed-prec requant | NO-GO Type-1 | Uniform Q3 worse every quality axis; local RMSE already rejects non-uniform |
| QTIP 3-bit quality | leaning NO-GO Type-2 | bits_needed both +; codec not run; decode already Type-1 dead |
| GLM-5.2 sub-bit MoE expert path | NO-GO Type-1 | Four families 0.116–0.157 cos @0.75 BPW; none beat null 0.898; student dense map works at 0.0104 BPW (not weight-space) |

**Axis closed for tps:** Q3_K, low-rank residual, learned codebook, QTIP Metal decode. Dense-tps routes to fusion/spec/stateful, not sub-Q4 bytes.

## Pre-spawn checklist

1. Is lever here? Read resurrection.
2. Measured cost share ≥ ship gate?
3. Calibration insight required? Run analyzer on retained artifacts only.
4. Downstream regressing (spec/batched verify)? Fix first.

## Cross-refs (abbrev)

`[[v110-path30-findings]]` `[[v230-icb-dead]]` `[[bible-execution-2026-05-30]]` `[[overnight-haul-2026-05-31]]` `[[moat-status-forward-path-2026-05-31]]` `[[silicon-solutions-2026-05-29]]` `[[glm52-pq-expert-function-bound-2026-07-23]]` composition-decision-matrix; gpu-us-accuracy; qtip/sub4bit oracles.
