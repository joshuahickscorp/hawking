# Ascension Family Kernel Architecture Plan

**Status:** PLAN ONLY — future programme, gated on Proto-Frankenstein offload.  
**Bible:** HAWKING_ASCENSION_BIBLE §29 (Family kernel architecture), with §30–§31, §32.  
**Companion plans:** `ASCENSION_MODEL_LADDER_PLAN.md`, `ASCENSION_ROTATION_RULE_PLAN.md`  
**Scaffold seeds already present (do not re-invent vocabulary):**

- `lab/operators/ascension_parity_ladder.py` — `ModelFamily`, `FAMILY_KERNEL_RECEIPT_SCHEMA`, stage inventories
- `lab/operators/research_registry.py` / `ASCENSION_RESEARCH_REGISTRY_PLAN.md` — `ADMIT_*` / `DEFER` / `REJECT`
- `crates/hawking-core/src/broker_kernel_ab.rs` — parity-then-speed promotion gate
- Tonight’s exact-model campaign on family key **`DEEPSEEK_V4`** (DeepSeek-V4-Flash / DSV4F)

**Explicit non-goals for this revision:** no Qwen/Gravity downloads, no live kernel
promotion, no edits under `lab/operators/frankenstein_*`, no mutation of DSV4F
crate authority paths (read-only reference), no push/PR/remote.

---

## 1. Purpose

After **exact-model success** on a family member, generalize into:

```text
shared semantic runtime
architecture-family execution graphs
generated geometry variants
exact-model fast paths where justified
```

A megakernel may be **one Metal function** or **one fused replayable command graph**.  
Promotion is only by **complete wall time** and **p99** (never GPU-only, never
dispatch-count alone — bible §29 and tonight’s P4A topology rule).

Tonight already *is* a deep exact-model implementation for `DEEPSEEK_V4`. The
family question this section asks — **which kernel grammar transfers to the next
family** — therefore has real measured data, not speculation. This plan freezes
that classification and designs the selection-key schema so later families reuse
the same machinery.

---

## 2. Initial families (bible §29)

| Family key | Bootstrap / flagship role | Architectural distinction |
|------------|---------------------------|---------------------------|
| `QWEN3_MOE` | 30B executor (bible §8) | top-8 / shared expert / standard GQA-or-MHA |
| `QWEN3_NEXT` | 80B reviewer (bible §9) | hybrid 3×DeltaNet + 1×gated attention; top-10 / 512 |
| `DEEPSEEK_V4` | Terra exact-model lab (tonight) | mHC, sparse attn ratios, FP4/FP8 native, top-6 |
| `LLAMA` | dense portable contract | no router; simpler residual graph |
| `MISTRAL_MIXTRAL` | sparse MoE (different gate) | Mixtral-style experts; shared MoE pack candidate |
| `STATE_SPACE_HYBRID` | long-context state machines | SSM / hybrid slots beyond DeltaNet |

Enum already scaffolded: `ModelFamily` in `ascension_parity_ladder.py`.

---

## 3. Four layers of the architecture

### 3.1 Shared semantic runtime

Family-agnostic contracts that must not fork per model:

| Contract | Tonight reference | Transfers to |
|----------|-------------------|--------------|
| NumericParity V2.1 | `hawking.numeric_parity.v2_1` on every sealed DSV4F receipt | all families |
| `claim_boundary` honesty | every DSV4F receipt field set | all |
| Status vocabulary | `PASS_FULL_STACK`, `PASS_NUMERIC_V2_1_ONLY`, `REJECT_*`, `*_WITHHELD` | all |
| Research admission | `ADMIT_TO_GRAVITY` / `ADMIT_TO_RUNTIME` / `ADMIT_TO_KERNEL` / `DEFER` / `REJECT` | all |
| Complete-token profiler stages | `_COMPLETE_TOKEN_PROFILE_STAGES` → `FAMILY_STAGES` | parameterized |
| Physical trace identity | `command_buffers`, `cpu_visible_waits`, `metal_dispatches`, `host_intermediate_handoff` | all Metal paths |
| Broker A/B gate | `broker_kernel_ab` — parity first, then wall/p99; never auto-serve-promote | all |

### 3.2 Architecture-family execution graphs

One **execution graph template** per family, not per weight revision:

```text
family_graph = ordered stage list
            + legal fusion boundaries
            + host-sync points that must remain
            + optional concurrent groups (expert waves)
            + state objects (KV, mHC, DeltaNet, SSM)
```

Examples from tonight:

| Graph fragment | DSV4F evidence | Family owner |
|----------------|----------------|--------------|
| P3A→P4A attention chain | P4B reseal stage profile (33 authority stages) | `DEEPSEEK_V4` |
| P6 MoE `dispatch_batch` (gate+route+up / down+combine) | `gravity_deepseek_v4_p6_device.rs`; L0–L1 receipt: **60 dispatches / 4 CBs** per MoE stage | MoE families (geometry params differ) |
| P7 mHC FFN pre/norm/post | multi-layer receipts `mhc_control_exp=darwin_double_double_control_domain_general` | `DEEPSEEK_V4` only |
| Learned-bias route (layers ≥3) | `dsv4f_learned_bias_route_metal_receipt.json` | `DEEPSEEK_V4` |

### 3.3 Generated geometry variants

Same kernel **grammar**, different:

```text
hidden, heads, experts, top_k, quant block, TG size, split-K, rows/TG
```

Tonight’s pattern: candidate symbols named by geometry
(`*_simdgroup_v4_splitk_candidate`, gate C1–C7, act-quant thread ladder),
selected by sealed A/B, never by vibes.

### 3.4 Exact-model fast paths

Where a geometry is so hot that a specialized Metal function beats the generated
variant **on complete wall + p99 with parity**, pin it as an exact-model fast
path. Fast paths are **not** the default family megakernel; they are justified
exceptions with a freeze receipt (bible §31: freeze grammar by geometry before
eviction).

---

## 4. Kernel selection key schema

Bible §29 lists the axes. Make them a **typed selection key**, not free prose.

### 4.1 Schema

```text
schema: hawking.ascension.family_kernel_selection_key.v1
```

| Field | Type | Notes |
|-------|------|-------|
| `family` | `ModelFamily` enum | `DEEPSEEK_V4`, `QWEN3_MOE`, … |
| `operator_grammar` | string enum | e.g. `gate_bf16_matvec_reduce`, `act_quant_bf16_ue8m0`, `fp4_e2m1_matvec`, `mhc_control_exp`, `command_topology_fuse`, `moe_gather_combine` |
| `tensor_dimensions` | object | rows, cols, packed_k, block, heads, … |
| `representation` | string | `bf16`, `fp8_e4m3fn_e8m0`, `fp4_e2m1fn_x2_e8m0`, `q4_k`, … |
| `active_experts` | object | `n_routed`, `top_k`, `shared_expert: bool` |
| `batch_session_count` | object | `batch`, `sessions`, `microbatch` |
| `context_regime` | string enum | `bos`, `decode_pos1`, `prefill_L`, `long_ctx`, `fullseq_capture` |
| `device_generation` | string | e.g. `apple_m3_ultra` |
| `memory_pressure` | enum | `LOW` / `MED` / `HIGH` / `STREAMING_WEIGHTS` |
| `megakernel_kind` | enum | `METAL_FUNCTION` / `FUSED_REPLAYABLE_COMMAND_GRAPH` |
| `parity_class` | `ParityClassification` | `NUMERIC_PARITY_V2_1_ONLY` / `EXACT_STORAGE` / … |
| `promotion_metric` | object | `{ wall_p50, wall_p99, gpu_p50_optional }` — wall+p99 required |

### 4.2 Receipt for a selected megakernel

```text
schema: hawking.ascension.family_kernel_selection_receipt.v1
```

Required fields:

```text
selection_key          (v1 object above)
candidates[]           each with kernel id, geometry, parity_pass, wall_p50/p99
winner                 or null
promotion_verdict      REJECT_PARITY | REJECT_NO_WIN | CANDIDATE_READY | SERVE_PROMOTED
                         (SERVE_PROMOTED only from protected controller — never sandbox)
claim_boundary         default_claim_boundary() extended
physical_trace         command_buffers, dispatches, cpu_visible_waits
transfer_class         FAMILY_SPECIFIC | FAMILY_GRAMMAR | CROSS_FAMILY_TOPOLOGY | PROCESS_ONLY
predecessor_receipts[] seal_sha256 list
status                 PASS_* / REJECT_* vocabulary (no new synonym system)
```

Scaffold placeholder already reserved: `FAMILY_KERNEL_RECEIPT_SCHEMA =
"hawking.ascension.family_kernel_plan.v1"` — promote to the two schemas above
when implementation lands; do not invent a third naming style.

### 4.3 Selection algorithm (honest)

```text
1. Filter by family + operator_grammar + representation + tensor_dimensions
2. Filter by parity_pass (V2.1 or exact-storage as required by rung)
3. Filter by device_generation + memory_pressure feasibility
4. Rank by complete host wall p50, then p99 (must not regress)
5. Optional secondary: GPU p50 only as diagnostic, never sole promote signal
6. Emit CANDIDATE_READY; controller alone may SERVE_PROMOTE
```

Matches tonight’s P4A topology promotion rule:

> candidate host-wall p50 must improve and candidate host-wall p99 must not regress  
> (`DSV4F_P4A_LAYER0_ATTENTION_TOPOLOGY_SWEEP-v1.json`)

---

## 5. Tonight’s DSV4F kernel wins — transfer classification

Sources are sealed or worktree-local measurement receipts. **No win is claimed
beyond its `claim_boundary`.**

### 5.1 Classification table

| Win | Evidence | Transfer class | Transfers as | Does **not** transfer as |
|-----|----------|----------------|--------------|---------------------------|
| **Sequence sharding** 1.40×@2w / 1.79×@4w | `evidence/parallelism/FULLSEQ_CAPTURE_PARALLELISM_FINDINGS.json` (`PASS_SHARDING_IMPLEMENTED_AND_MEASURED`; bit-exact merge) | **CROSS_FAMILY_TOPOLOGY** | Multi-process sequence partition + offline merge whenever **host serial orchestration** is the bottleneck and residuals do not cross workers | Layer sharding without residual handoff; stream/link-bound capture (GLM official path Amdahl ~1.05× — see `evidence/parallelism/GLM_TEACHER_FORCED_PARALLELISM_FINDINGS.json`) |
| **Command-buffer collapse** 21 CB → **1 CB** (same 21 dispatches) | `DSV4F_P4A_LAYER0_ATTENTION_TOPOLOGY_SWEEP-v1.json` — host wall p50 **62579→54271 µs**, p99 **63527→54510 µs**, `promoted: true` | **CROSS_FAMILY_TOPOLOGY** | Ordered multi-encoder single-commit graphs; `dispatch_batch` wait collapse | Unproven same-encoder reordering; claiming GPU FLOPS win when GPU µs was already flat (~54 ms) |
| **P6 MoE batching** 60 dispatches / **4 CBs** | L0–L1 multi-layer receipt stages `l*_p7_mhc_ffn_p6_moe_mhc_post`; `gravity_deepseek_v4_p6_device.rs` | **FAMILY_GRAMMAR** (MoE) | Expert-wave concurrent groups + batch commits for any MoE family | DSV4 FP4 packing and hash/learned-bias route math |
| **Gate reduction C4 simd32** | `DSV4F_P0_GATE_REDUCTION_SWEEP-v1.json` — C4 `candidate_vs_live_fp64_pass=true`, `promotion_eligible=true`, GPU p50 **47 µs**; receipt still `receipt_promoted=false` | **FAMILY_GRAMMAR** | simdgroup-width=32 FMA reduction association for gate-like matvecs | Exact 256×4096 BF16 DSV4 gate tensor; auto-promote without P6 integration |
| **Act-quant SIMDgroup block winner** | `DSV4F_ACT_QUANT_SIMDGROUP_SWEEP-v1.json` — winner GPU p50 **95 µs** vs fixed authority **5967 µs** (~**62.8×**); `MODEL_LINEAR_COMPONENT_QAT_CANDIDATE_ONLY` | **FAMILY_GRAMMAR** (low-prec glue) | SIMDgroup block quant before FP4/FP8 matvec | Serving path until full residual compose + parity |
| **Raw-weight SIMDgroup split-K** | `DSV4F_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP-CANONICAL-v1.json` — FP4 **24.3×**, FP8 **30.0×** GPU p50 vs serial authority; **NOT_PROMOTED** | **FAMILY_GRAMMAR** | split-K + simdgroup candidate ladder for bandwidth-bound matvec | Source-native FP4/FP8 layouts outside DeepSeek packing; serve flip |
| **mHC Darwin double-double exp** | `deepseek_v4_mhc_control_exp.metal`; P4B reseal `…-reseal-darwin-dd.json`; multi-layer `mhc_control_exp=darwin_double_double_control_domain_general` | **FAMILY_SPECIFIC** math; **PROCESS** for the failure class | Pattern: host transcendental mismatch → reconstruct host lib on Metal; control-domain exactness before residual V2.1 | mHC Sinkhorn / sigmoid / HC mix equations; any non-DeepSeek architecture without mHC |
| **NumericParityV21Only + claim_boundary** | All multi-layer receipts; P4B reseal | **PROCESS_ONLY** | Every family ladder | Treating V2.1 as exact-storage |
| **Physical-trace counters** | fullseq findings; multi-layer metal blocks | **CROSS_FAMILY_TOPOLOGY** | Diagnosis of host-serial vs GPU-bound | Inferring occupancy from absent Metal APIs |

### 5.2 Command-buffer collapse — detail

| Topology | CBs | Dispatches | Waits | Host wall p50 |
|----------|-----|------------|-------|---------------|
| Baseline (per-stage authority) | 21 | 21 | 21 | 62579 µs |
| Candidate (one CB, ordered encoders) | **1** | 21 | **1** | **54271 µs** |

Still open after tonight: P4B position-1 complete attention reseal still records
**33 CBs / 33 dispatches** for the full causal path
(`command_topology.current` on the darwin-dd reseal) — the one-CB win is
**retained as predecessor** (`p4a_one_cb_topology_win`) and is the grammar to
generalize, not a finished full-decode graph.

Multi-layer compose today:

| Scope | CBs | Dispatches | Status |
|-------|-----|------------|--------|
| L0–L1 | 20 | 169 | `PASS_MULTI_LAYER_GPU_FORWARD_L0_L1` |
| L0–L2 | 26 | (higher) | `PASS_MULTI_LAYER_GPU_FORWARD_BOS_L0_L2` |
| L0–L42 | 265 | — | `PASS_MULTI_LAYER_GPU_FORWARD_BOS_L0_L42` |

Megakernel target: drive CBs toward **one fused replayable command graph per
token** where host handoff is zero (`host_intermediate_handoff_between_stages: false`
already true on L0–L1).

### 5.3 Sequence sharding — detail

| Workers | Wall s | Speedup | Efficiency |
|---------|--------|---------|------------|
| 1 | 9.186 | 1.00 | 1.00 |
| 2 | 6.531 | **1.40** | 0.70 |
| 4 | 5.106 | **1.79** | 0.45 |

Bit-exact npy + trace site hashes vs serial. Primary bottleneck named:
**~5 `cpu_visible_waits` per token×layer** on a single host core while GPU sat
in the 60–70% band. This is the textbook case for CROSS_FAMILY transfer of the
**orchestration** pattern, not of DeepSeek math.

### 5.4 Shared vs family-specific broker surface (from KERNEL_BROKERS_TUNING_PLAN)

**Tune once (shared):**

- MoE top-k gate / gather / route accumulate (geometry-parameterized)
- Act quant + scale blocks
- RMSNorm / SwiGLU / cast / rope
- KV append / multiseq scatter
- Sample / argmax
- CB / expert-wave topology
- NumericParity V2.1 harness

**Do not pretend shared:**

- FP4 E2M1×2 + E8M0 expert matvec (DeepSeek packing)
- FP8 E4M3FN control matvec at DSV4 shapes
- mHC pre/post + Darwin DD control exp
- Hash tid2eid route + learned-bias sqrtsoftplus
- Sparse attention ratio-0/4/128 indexer

---

## 6. Megakernel promotion law

```text
parity_pass
AND complete_wall_p50 improved
AND complete_wall_p99 not regressed
AND fallback_count == 0
AND real GPU dispatches > 0
AND claim_boundary respected
→ CANDIDATE_READY
```

Only protected controller / human may:

```text
CANDIDATE_READY → SERVE_PROMOTED
```

Never:

- GPU p50 alone
- fewer waits with worse wall (graveyard class: *fewer waits but slower wall time*)
- component win as full-stack claim
- sandbox self-promotion (`ForbiddenAuthoritativeClaim`)

---

## 7. Implementation sequence (after Proto-Frankenstein offload)

| Step | Action | Gate |
|------|--------|------|
| 0 | Land selection-key + selection-receipt types next to `ascension_parity_ladder` | unit tests only |
| 1 | Ingest tonight’s DSV4F wins into a `KernelGrammarTransferLedger` with `transfer_class` | read-only receipts |
| 2 | Exact-model finish line for DSV4F residual + capability (`PASS_FULL_STACK` or honest withhold) | §8-style ladder |
| 3 | Extract CROSS_FAMILY_TOPOLOGY grammars (CB fuse, sequence shard, physical trace) into shared runtime | parity holds on DSV4 |
| 4 | Extract FAMILY_GRAMMAR MoE pack for `QWEN3_MOE` geometry | 30B stream after offload |
| 5 | Generate geometry variants; keep exact-model fast paths only where wall+p99 justify | A/B harness |
| 6 | Family megakernel candidates for Qwen; TG gauntlet | TG plan |
| 7 | At TG3 stop for human review before family-wide promote | rotation rule |

---

## 8. Receipt & type inventory (this plan owns)

| Schema | Role |
|--------|------|
| `hawking.ascension.family_kernel_selection_key.v1` | lookup key |
| `hawking.ascension.family_kernel_selection_receipt.v1` | sealed choice |
| `hawking.ascension.kernel_grammar_transfer_entry.v1` | one win + transfer_class |
| `hawking.ascension.kernel_grammar_transfer_ledger.v1` | ledger of entries |
| `hawking.ascension.megakernel_candidate.v1` | Metal function **or** fused command graph descriptor |

Reuse existing:

| Existing | Role |
|----------|------|
| `hawking.numeric_parity.v2_1` | parity |
| `hawking.ascension.parity_ladder_receipt.v1` | family ladder |
| `hawking.lab.candidate_report.v1` / `ROADBLOCK_CANDIDATE` | report-only authority |
| Research registry five-way admit | technique admission |

---

## 9. Honesty / non-claims

- DSV4F is **not** yet a full HCLI serve path; many receipts explicitly deny
  `BASE_TRUE_TPS`, full 43-layer runtime, or `PASS_FULL_STACK`.
- SIMDgroup / split-K / act-quant wins are **component candidates**, mostly
  `NOT_PROMOTED` into serve.
- Gate C4 is **eligible within probe**, not receipt-promoted into P6.
- Sequence sharding speedups were measured **under concurrent live L1** — lower
  bounds; remeasure idle before production roll-out.
- This document does not authorize family megakernel construction before
  Proto-Frankenstein offload and exact-model gates for the bootstrap Qwen pair.

---

## 10. Success criteria for §29

```text
[ ] Selection key schema implemented and round-tripped
[ ] Transfer ledger contains tonight’s DSV4F wins with transfer_class
[ ] At least one CROSS_FAMILY_TOPOLOGY grammar reused on a second family with sealed parity
[ ] Megakernel promote uses wall p50 + p99 only
[ ] Exact-model fast paths listed with freeze-by-geometry receipts before any source eviction
```
