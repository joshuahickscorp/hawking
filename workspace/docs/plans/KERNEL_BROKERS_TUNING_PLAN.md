# Kernel Brokers Tuning Plan

**Status:** executable plan + safe groundwork only  
**Date:** 2026-08-05  
**Scope:** Gravity kernels that serve the main open-source-model brokers  
**Non-goals:** no edits to `gravity_deepseek_v4*.rs` forward lane; no sealed-artifact mutation; no TPS claims without clean-room measurement; no auto-promotion of candidates.

## Brokers (serving priority)

| Tier | Model | Role | Geometry that drives kernels |
|---|---|---|---|
| **Terra** | DeepSeek-V4-Flash Gravity | primary HCLI body | 43 layers, H=4096, 256 routed / 6 active / 1 shared, expert inter=2048, FP4 experts + FP8 control |
| **Luna** | Qwen3-Coder-30B-A3B Gravity | primary code broker | 128 experts / 8 active / 0 shared (Qwen3-MoE shape); 256K native ctx → KV path matters more than Terra |
| **Frankenstein** | composite body | transfer / fusion body | reuses Terra/Luna kernel inventory; composition is model-specific, matvecs are not |

**Correctness posture today:** multi-layer DeepSeek-V4 GPU forward is sealed at `NUMERIC_PARITY_V2_1_ONLY` (not exact-storage e2e). Authority kernels are serial, correctness-first, **not** speed-tuned. Speed work must re-earn same-model before/after parity before any serve-path flip.

---

## Evidence base (real numbers only)

| Source | What it measures | Device |
|---|---|---|
| `receipts/dsv4f_multi_layer_gpu_forward_l0_l1_receipt.json` | L0–L1 full Metal forward wall + dispatch/CB topology | (run host) |
| `receipts/dsv4f_multi_layer_gpu_forward_bos_l0_l2_receipt.json` | L0–L2 full + L3 attention-only; 276 dispatches / 26 CBs | (run host) |
| `receipts/DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v1-reseal-darwin-dd.json` | Per-kernel GPU/CPU µs for complete position-1 attention (33 stages) | Apple M3 Ultra |
| `receipts/dsv4f_learned_bias_route_metal_receipt.json` | Learned-bias route kernel seal (1 dispatch) | (run host) |
| `workspace/campaign/evidence/models/deepseek-v4/DEEPSEEK_V4_RUNTIME_ACCOUNTING.json` | Analytical served roofline (not wall TPS) | M3 Ultra BW 417.7 GB/s |
| P6 execute graph in `gravity_deepseek_v4_p6_device.rs` (read-only) | Exact 60-dispatch MoE inventory matching receipts | — |
| `tools/bench/broker_kernel_ab/receipt_costs.json` | Machine-readable extract of the above | — |

### Layer wall (correctness-first serial-authority path)

| Receipt | Wall | Topology | Derived cost |
|---|---|---|---|
| L0–L1 full | **17 984.4 ms** | 169 dispatches / 20 CBs / 20 waits | **≈ 8.99 s/layer** |
| L0–L2 full + L3 attn | **22 901.1 ms** | 276 dispatches / 26 CBs / 26 waits | ≈ 7.6 s/layer if attributing 3 full layers (matches campaign “~7.6 s/layer”); L3 MoE refused (learned-bias not composed) |

Parity on both: `NUMERIC_PARITY_V2_1_ONLY`. No greedy token. No serve-endpoint flip.

### Per full MoE layer — dispatch inventory (receipt-matched)

Each full layer’s P6+P7 stage is **63 dispatches / 4 command buffers** (`p6_dispatches=60`, `p7_owned_dispatches=3`). Attention stages are ~21–22 dispatches (L0 growing-KV used 11 CBs; later BOS windows collapse to 1 CB).

| Family | Dispatches / full MoE layer | Notes |
|---|---:|---|
| **FP4 expert matvec** (`deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority`) | **18** | 6 experts × (W1+W3+W2); concurrent within CB groups; TG=256 authority |
| **FP8 shared matvec** (`deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority`) | **3** | shared W1/W3/W2; TG=256 |
| **Act quant** (`deepseek_v4_act_quant_bf16_ue8m0_authority`) | **8** | 1 ordered input + 7 concurrent (6 routed + 1 shared down); TG=32 authority; sealed component authority GPU **5 967 µs** (`FIXED_AUTHORITY_GPU_US`) |
| BF16 cast | 21 | glue; cheap GPU, not free in dispatch tax |
| SwiGLU (routed+shared) | 7 | includes route-weight scale on routed path |
| Gate + route | 2 | Gate is BF16[256,4096] matvec; route 1-thread |
| Combine | 1 | 6-routed + shared gather |
| **mHC control (P7)** | **3** | pre (1-thread authority) + ffn rmsnorm + post (TG=256) |

**Honesty gap:** multi-layer receipts do **not** break out per-kernel GPU µs for the MoE stage. Ranking below uses (dispatch count × shape bytes × authority geometry) plus the P4B attention micro-profile where available. Filling MoE per-kernel GPU µs is the first harness job.

### P4B attention micro-profile (position-1 complete, M3 Ultra)

Total GPU **101.55 ms** / CPU wall **116.56 ms** over **33 serial CBs** (1 dispatch per CB — correctness topology, not a decode graph).

| Rank (GPU) | Kernel | Disp. | GPU ms | CPU ms | Bytes read |
|---:|---|---:|---:|---:|---:|
| 1 | `deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority` (**mHC attn pre**) | 2 | **76.61** | 82.52 | 3.2 MB |
| 2 | `deepseek_v4_act_quant_bf16_ue8m0_authority` | 5 | **17.46** | 18.95 | 43 KB |
| 3 | `deepseek_v4_p4a_kv_nonrope_qat_inplace_authority` | 2 | 3.41 | 3.96 | 2 KB |
| 4 | `deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority` | 5 | **1.67** | 3.65 | **75.5 MB** |
| 5 | `deepseek_v4_p3a_rmsnorm_bf16_authority` | 5 | 1.53 | 3.02 | 41 KB |
| 6 | `deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority` | 1 | 0.62 | 1.18 | 33.6 MB |
| … | rope / sparse sink / **KV cache write** / mHC post | — | ≤0.10 each | — | tiny |

Interpretation for brokers:

- **mHC pre is the attention-side wall**, not sparse attention itself (0.10 ms) or KV write (0.01 ms).
- **Act quant is disproportionately expensive** for its tiny traffic (serial authority, TG=32).
- **FP8 control matvec is bandwidth-heavy but GPU-cheap** at these shapes on M3 Ultra (~0.3 ms/dispatch while reading tens of MB) — still multiplies across every projection every layer.
- **KV write is not the broker bottleneck at BOS/pos1**; Luna’s 256K ctx will invert that once long-context decode is live.

### Analytical roofline (not wall TPS)

From `DEEPSEEK_V4_RUNTIME_ACCOUNTING.json` (served weights resident, M3 Ultra 417.7 GB/s):

- Active bytes/token ≈ 247 MB (attention 145 MB + MoE 103 MB)
- Roofline ceiling: functional **54.0 TPS**, teacher **35.7 TPS**
- Explicitly **not** a measured wall-clock serve number; streamed forward is weight-reload bound.

---

## Kernel ranking by broker impact

Priority = (serve-path frequency) × (per-call cost or dispatch mass) × (shared-broker leverage).  
Costs are **current authority / receipt numbers**, not targets.

### 1. FP4 expert matvec — **P0 (Terra dominant; Luna equivalent once native quant lands)**

| Field | Value |
|---|---|
| Authority kernel | `deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority` |
| Current cost | **18 dispatches / full MoE layer** (60% of P6). Shapes: W1/W3 rows=2048 packed_k=2048; W2 rows=4096 packed_k=1024. Authority TG=**256**, one-row serial reduce (`#pragma clang fp contract(off)`). Multi-layer wall ≈ **9 s/layer** is MoE-dominated; no sealed per-dispatch GPU µs yet. |
| Serve impact | Terra: every token, every layer, 6 experts × 3 projs. Luna: same structural role with Q4_K / future native expert quant via `moe_grouped_gemm_*`. Frankenstein: reuses. |
| Tuning levers | (1) **SIMDgroup width + split-K** candidates already scaffolded (`*_simdgroup_v4_splitk_candidate` pattern for FP8; port to FP4). (2) **Packed loads** of E2M1×2 + E8M0 scale tiles into threadgroup. (3) **Codebook / scale caching** in TG memory for the 32-elem FP4 block. (4) **Rows-per-threadgroup** sweep (existing component + split-K ladders: 32…1024). (5) Keep **expert-wave concurrency** (already `begin_concurrent_group` for W1/W3 and W2 waves) but collapse host waits further. |
| Parity gate | Same sealed bytes + FP64 / NumericParity V2.1 vs authority; exact top-k route IDs unchanged. |

### 2. Command-buffer / wait collapse (topology) — **P0 (shared tax on all brokers)**

| Field | Value |
|---|---|
| Current cost | L0–L1: **20 CBs / 20 cpu-visible waits** for 169 dispatches; L0–L2: **26/26** for 276. P4B attention alone: **33 CBs for 33 dispatches**. P6 already batches to **4 CBs / 60 dispatches** — better, still 4 waits/layer for MoE alone. |
| Serve impact | Every broker token pays encode+submit+wait. At ~9 s/layer wall with ~100 ms of pure attention GPU, **topology and residency dominate wall**. |
| Tuning levers | Persistent **single decode CB** / multi-encoder graph; pipeline precompile (already true on P4B); eliminate empty CBs; fuse stage boundaries that today force batch commits; reuse GLM `GPU_EXPERT_WAVE` wait-collapse pattern for DSV4/Qwen. |
| Parity gate | Physical-trace identity + zero host intermediate handoff (receipt field already tracks this). |

### 3. Act quant (BF16→UE8M0) — **P0 (shared glue before every FP4/FP8 matvec)**

| Field | Value |
|---|---|
| Authority kernel | `deepseek_v4_act_quant_bf16_ue8m0_authority` |
| Current cost | P4B: **17.46 ms GPU / 5 dispatches** (~3.5 ms each) despite ~8 KB read. Component seal: **5 967 µs** fixed authority GPU stage. P6: **8 dispatches/layer**. Candidate ladder exists: `deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate`. |
| Serve impact | Multiplies with every projection (attention + MoE). Highest GPU-ms-per-byte in the sealed attention profile. |
| Tuning levers | SIMDgroup block candidate promotion **only with byte-exact act+scale hashes**; TG width ladder 32→1024 (sweep example already written); fuse quant into following matvec where association order allows. |
| Parity gate | Byte-exact activation + scale SHA-256 vs sealed oracle (act_quant sweep contract). |

### 4. FP8 control matvec — **P1 (Terra attention + shared expert; Luna dense/control path)**

| Field | Value |
|---|---|
| Authority kernel | `deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority` |
| Current cost | P4B: **1.67 ms GPU / 5 dispatches**, **75.5 MB** read. P6 shared: 3/layer. Authority TG=256. Split-K candidate: `…_simdgroup_v4_splitk_candidate`. |
| Serve impact | All attention projections (wq_a/b, wkv, wo_b) + shared expert. Bandwidth-bound; roofline shows attention owns ~58% of token bytes. |
| Tuning levers | Split-K + SIMDgroup v4 candidate (already in `matmul.metal`); packed FP8 loads; dual-issue with act_quant fusion; keep concurrent group with FP4 wave where deps allow. |
| Parity gate | NumericParity V2.1 op-local bounds vs FP64; storage exact on QAT inputs. |

### 5. mHC control path — **P1 (Terra/Frankenstein; not Qwen)**

| Field | Value |
|---|---|
| Kernels | Attn pre: `deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority`; FFN: `deepseek_v4_p7_mhc_ffn_pre_authority`, `…_ffn_rmsnorm_bf16_authority`, `…_mhc_ffn_post_authority`; control exp: `deepseek_v4_mhc_control_exp.metal` / Darwin DD candidates |
| Current cost | Attn pre alone: **76.61 ms GPU / 2 dispatches** in P4B (75% of attention GPU). P7: 3 dispatches/layer; pre+norm use **1-thread** authority geometry. Multi-layer notes `mhc_control_exp=darwin_double_double_control_domain_general`. |
| Serve impact | Every Terra layer, both attn and FFN sides. Dominates attention micro-profile. |
| Tuning levers | Parallelize Sinkhorn / mix beyond 1-thread **only with DD-compatible exp**; promote darwin-dd candidates that already exist as traces; do **not** trade control-domain math for speed. |
| Parity gate | Control-domain exactness first; NumericParity V2.1 on residual outputs. |

### 6. MoE gate + route — **P1 (both brokers; mode differs)**

| Field | Value |
|---|---|
| Kernels | Gate: `deepseek_v4_p6a_gate_bf16_matvec_authority` / C4 simd32 (`P6_C4_GATE`); hash route: `deepseek_v4_p6a_hash_route_sqrtsoftplus_authority`; learned bias: `deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority` |
| Current cost | 2 dispatches/layer (gate+route) inside P6. Gate = BF16 matvec 256×4096 with **exact simdgroup=32**. Learned-bias route sealed separately: 1 dispatch, receipt wall 6.5 s includes setup (not steady-state). L3+ full MoE **blocked** until learned-bias two-phase compose lands. Shared `moe_topk_gate` for Qwen/Mixtral shapes. |
| Serve impact | Wrong route = wrong experts = silent quality death. Gate reduction association already has C1–C7 candidates under sweep. |
| Tuning levers | Gate: promote C4 simd32 only if reduction-association receipt passes; Route: device-side learned bias (kernel sealed) → compose two-phase expert load; Qwen: `moe_topk_gate` top-8/128 with threadgroup logits. |
| Parity gate | **Exact expert IDs** (hard, no tolerance) + V2.1 on scores/weights. |

### 7. Expert gather / combine + expert-wave concurrency — **P1 (shared structure)**

| Field | Value |
|---|---|
| Kernels | `deepseek_v4_p6a_route6_shared_combine_bf16_authority`; shared `moe_gather_combine` / `moe_route_accumulate`; GLM expert-wave flag pattern |
| Current cost | 1 combine dispatch/layer; expert wave already concurrent for projections but **not** wait-collapsed to 1 drain/layer (GLM estimates show large wait wins when wave is on). |
| Serve impact | Every MoE token. Luna top-8 increases gather width vs Terra top-6. |
| Tuning levers | Weighted gather fusion into W2 epilogue; expert-wave wait collapse (GLM `HAWKING_GLM_GPU_EXPERT_WAVE` lesson); optional concurrent projection groups behind explicit flag. |
| Parity gate | Exact BF16/F32 combine order vs source; route weights applied once. |

### 8. KV read/write — **P2 today / P0 for Luna long-context**

| Field | Value |
|---|---|
| Kernels | `deepseek_v4_p4b_kv_cache_write_bf16_authority`; sparse sink `deepseek_v4_p4b_sparse_attention_position1_two_kv_sink_authority`; growing-KV ratio-0 / ratio-4 / ratio-128 paths; generic `kv_scatter_append_multiseq` |
| Current cost | Pos1 write: **0.01 ms GPU**; two-KV sparse sink: **0.10 ms**. Bytes tiny at BOS. Runtime accounting still assigns **attention 145 MB/token** active (includes projections + cache traffic at length). |
| Serve impact | Terra short decode: minor. **Luna 256K**: becomes the broker wall; multi-seq scatter path is the shared win. |
| Tuning levers | Fused rope+KV append (existing fused kernels in `common.metal`); paged / windowed KV for sparse ratios; INT4/F16 KV variants already have parity tests in tree — promote only with sealed decode parity. |
| Parity gate | Exact KV row storage where claimed; V2.1 on attention outputs. |

### 9. lm_head + sampling — **P2 (once full forward produces logits)**

| Field | Value |
|---|---|
| Kernels | `gemv_*` / GLM native bf16 lm_head; `sample_argmax_f32`, temperature/top-k/top-p/multinomial in `sample.metal` |
| Current cost | Multi-layer receipts: **`greedy_token_produced: false`**. Historical note in `numeric_parity.rs`: V2 lm_head gate was mis-conditioned; V2.1 is the authority. No sealed DSV4 lm_head GPU µs in this worktree’s receipts. |
| Serve impact | One matvec/token × vocab; must be correct for HCLI. Not on the critical path until L0–L43 residual chain + head land. |
| Tuning levers | SIMDmat lm_head (existing `v1x_lm_head_simdmat_parity`); device argmax; vocab prune when policy allows. |
| Parity gate | V2.1 full-forward logits bounds + **exact greedy argmax**. |

---

## Shared vs model-specific kernels

### Shared across DeepSeek (Terra) + Qwen (Luna) brokers — **tune once**

| Capability | Shared surface | Notes |
|---|---|---|
| MoE top-k gate / gather | `moe.metal`: `moe_topk_gate`, `moe_gather_combine`, `moe_route_accumulate`, grouped GEMM | Geometry-parameterized (n_experts, top_k) |
| Act quant + scale blocks | `matmul.metal` act_quant authority + candidates | Same glue before low-precision matvec |
| RMSNorm / SwiGLU / cast / rope | `common.metal` | Universal |
| KV append / multiseq scatter | `kv_scatter_append_multiseq`, rope+append fusions | Critical for Luna 256K and any multi-seq serve |
| Sample / argmax | `sample.metal` | End of every broker |
| CB / expert-wave topology | Metal batching + GLM expert-wave pattern | Wait collapse is arch-agnostic |
| Q4_K / quant matvec | `quant.metal`, `moe_grouped_gemm_q4` | Luna near-term; Terra secondary if containers re-quant |
| NumericParity V2.1 + harness gate | `numeric_parity` + `broker_kernel_ab` | Promotion discipline shared |

### Model-specific — **do not pretend they are shared**

| Kernel family | Owner | Why specific |
|---|---|---|
| FP4 E2M1×2 + E8M0 expert matvec | Terra (DeepSeek-V4) | Source-native packing + scale association |
| FP8 E4M3FN + E8M0 control matvec (DSV4 shapes) | Terra | Source linear layout / QAT blocks |
| mHC pre/post + Darwin DD control exp | Terra / Frankenstein | Architecture-unique |
| Hash tid2eid route + sqrtsoftplus | Terra layers 0–2 | Source routing table |
| Learned-bias route (sqrtsoftplus + bias) | Terra layers ≥3 | Sealed kernel; P6 two-phase **not composed** |
| Sparse attention ratio-0/4/128 + indexer | Terra | DeepSeek sparse sink |
| Qwen3-MoE top-8/128, no shared expert | Luna | `qwen_moe.rs` still Phase-0 stub; uses generic MoE pack, not DSV4 P6 |
| Frankenstein fusion schedule | Frankenstein lane | Transfer/composition only — **out of this plan’s edit scope** |

**Concentration rule:** spend the first tuning budget on **shared** items (act quant, CB collapse, MoE gate/gather, KV multiseq, sample). Spend the second on **Terra FP4 expert matvec** (largest dispatch mass on the live sealed path). Luna-native expert quant rides shared MoE pack once the Qwen3-Coder Gravity body is resident.

---

## Tuning program (executable sequence)

All steps are **candidate-only** until the broker A/B harness prints `parity_pass=true`. Scaffold never auto-promotes (`PromotionVerdict::RefusedManual` always for serve flip).

| Step | Action | Lever | Parity oracle | Collision risk |
|---|---|---|---|---|
| 0 | Land harness (this plan) | gate + cost registry | unit tests | none |
| 1 | Seal per-kernel GPU µs for one P6 MoE layer | profiling only | existing multi-layer receipt | none (read-only examples) |
| 2 | Act quant SIMDgroup candidate A/B | TG / simdgroup | byte-exact act+scale | low — new candidate shader names only |
| 3 | FP4 split-K / rows-per-TG sweep | SIMDgroup, split-K, packed loads | P5B/P6 component oracles | low if new `*_candidate` symbols |
| 4 | FP8 split-K candidate A/B | split-K | model.linear / P4B component | low |
| 5 | Gate reduction C4 promotion decision | simdgroup association | gate_reduction_sweep | medium — must not retune without receipt |
| 6 | Compose learned-bias route into P6 two-phase | route + expert load | learned_bias_route receipt + P6 | **HIGH — forward lane owns P6** |
| 7 | CB collapse for decode graph | command-buffer | physical trace + V2.1 | **HIGH — forward lane** |
| 8 | mHC pre parallelization under DD exp | control math | P4B/P7 oracles | **HIGH — math domain** |
| 9 | Luna Qwen3-Coder body + shared MoE pack | gather/gate | new Luna receipts | separate lane |
| 10 | lm_head + sample once logits exist | simdmat / argmax | V2.1 + exact argmax | after full residual |

Steps 6–8 intentionally wait on the DeepSeek-V4 forward lane; this plan only prepares oracles, costs, and the A/B gate.

---

## Safe groundwork delivered with this plan

| Path | Role |
|---|---|
| `workspace/docs/plans/KERNEL_BROKERS_TUNING_PLAN.md` | this document |
| `tools/bench/broker_kernel_ab/receipt_costs.json` | receipt-extracted costs |
| `tools/bench/broker_kernel_ab/README.md` | how to run A/B without promoting |
| `crates/hawking-core/src/broker_kernel_ab.rs` | pure gate + kernel registry (no forward-lane deps) |
| `crates/hawking-core/examples/broker_kernel_ab_harness.rs` | CLI scaffold: dry-run A/B, refuse promotion without parity |
| `crates/hawking-core/tests/broker_kernel_ab_gate.rs` | unit tests for the promotion gate |

**Promotion rule (enforced in code):**

```text
if !parity_pass:          verdict = RejectParity
elif !speed_improved:     verdict = RejectNoWin   (parity-only record is fine)
else:                     verdict = CandidateReady  // NEVER ServePromote
// ServePromote is not constructible from the scaffold API.
```

---

## Explicit non-claims

- No BASE_TRUE_TPS / HCLI serve numbers.
- No claim that split-K / SIMDgroup candidates beat authority until a sealed A/B receipt says so.
- No edit to `gravity_deepseek_v4*.rs`, `ramanujan/`, sealed artifacts, or Frankenstein transfer lanes.
- L3+ full MoE remains refused until learned-bias two-phase is composed on the forward lane.

## Next human / lane checkpoints

1. Forward lane: learned-bias P6 compose (unblocks ratio-128 full layers).  
2. Brokers lane: run act_quant + FP4 A/B through this harness against sealed oracles.  
3. Luna lane: admit Qwen3-Coder-30B-A3B Gravity body; wire shared MoE pack.  
4. Only after (2)+(3): clean-room coexist_bench for broker TPS — never from agent coexist noise.
