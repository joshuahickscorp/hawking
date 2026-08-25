# Ascension Rotation Rule Plan

**Status:** PLAN ONLY — future programme, gated on Proto-Frankenstein offload.  
**Bible:** HAWKING_ASCENSION_BIBLE §31 (Rotation rule), with §10 (TG), §29–§30, §32 (Graveyard).  
**Companion plans:** `ASCENSION_FAMILY_KERNEL_ARCHITECTURE_PLAN.md`, `ASCENSION_MODEL_LADDER_PLAN.md`, `ASCENSION_TG_GAUNTLET_HARNESS_PLAN.md`, `ASCENSION_GRAVEYARD_PLAN.md`  
**Scaffold seeds:**

- `lab/operators/ascension_parity_ladder.py` — `RotationTrigger`, rotation_rule text in pipeline receipt
- `lab/verification_authority.py` — `ROADBLOCK_CANDIDATE`, `TG_RUNG_CANDIDATE`, forbidden authoritative claims
- `lab/operators/ascension_graveyard.py` / graveyard plan — mechanism burial + reopen
- Tonight’s mHC ULP / Darwin double-double investigation (worked example)

**Explicit non-goals:** no live rotation of production models, no source eviction
in this revision, no edits to frankenstein operators or DSV4F authority crates.

---

## 1. Purpose

Rotate the active optimization subject when **either**:

```text
A. the model descends at least one named TG rung
```

or:

```text
B. two materially different optimization architectures fail,
   the same-model roofline is measured,
   the repeated bottleneck is sealed,
   and the smallest next representation change is named
```

At **TG3**: always stop for human/controller review.

Before source eviction: **freeze every winning kernel grammar by geometry**.

---

## 2. Trigger A — TG rung descent

### 2.1 Named rungs (from TG plan / bible)

| Rung | Target TPS |
|------|------------|
| TG32 | 31.25 |
| TG20 | 50 |
| TG16 | 62.5 |
| TG12 | 83.3 |
| TG10 | 100 |
| TG8 | 125 |
| TG5 | 200 |
| TG4 | 250 |
| **TG3** | **333** — **mandatory human/controller stop** |
| TG2 | 500 — human-authorized only after TG3 |
| TG1 | 1000 |

### 2.2 What “descends” means

```text
schema: hawking.ascension.tg_rung_descent_receipt.v1
```

| Field | Requirement |
|-------|-------------|
| `model_id` / `family` | same model as gauntlet |
| `from_rung` / `to_rung` | named rungs; `to` is lower TPS target (higher performance) by ≥1 step |
| `scoreboard` | `BASE_TRUE_TPS` only (never blended with accelerated/block scores) |
| `tg_rung_requirements` | same model, same capability tier, complete-token timing, batch-1 base, CLEAN, fallback=0, real GPU, stable p99, prompt-dependent coherent generation |
| `status` | `TG_RUNG_CANDIDATE` until controller certifies |
| `claim_boundary` | no HCLI workforce claim unless product tests pass |

**A alone is sufficient to rotate** after freeze+seal of winning grammars — except
when `to_rung` is TG3 or beyond: then Trigger A **pauses at REVIEW**.

### 2.3 TG3 hard stop

```text
on clear TG3:
  status = TG3_REVIEW_REQUIRED
  stop autonomous promotion
  checkpoint
  seal complete evidence
  notify human / protected controller
  do NOT auto-enter TG2/TG1
  do NOT auto-declare FAMILY_EXHAUSTED
```

Human/controller may then: promote workforce role, authorize representation
change, continue toward TG2/TG1, or open family megakernel generalization.

---

## 3. Trigger B — two architectures + roofline + sealed bottleneck

All four conjuncts required:

| # | Conjunct | Evidence form |
|---|----------|---------------|
| B1 | **Two materially different optimization architectures fail** | two sealed attempt receipts with distinct `architecture_class` |
| B2 | **Same-model roofline measured** | complete-token profile + bandwidth/FLOP ceilings; ≥98% wall named |
| B3 | **Repeated bottleneck sealed** | same `mechanism_class` / bottleneck id across attempts; graveyard or roadblock seal |
| B4 | **Smallest next representation change named** | explicit next change (quant layout, fusion boundary, state contract, packing) — not “try harder” |

```text
schema: hawking.ascension.rotation_trigger_b_receipt.v1
```

| Field | Notes |
|-------|-------|
| `model_id` / `family` | |
| `attempts[]` | ≥2; each has `architecture_class`, `mechanism_class`, `status`, `seal_sha256` |
| `material_difference_predicate` | why attempts are not three knobs on one idea |
| `roofline` | receipt ref |
| `repeated_bottleneck` | id + mechanism_class + sealed evidence |
| `next_representation_change` | named smallest change |
| `status` | `ROADBLOCK_CANDIDATE` → controller may certify `ROTATE` |
| `claim_boundary` | |

### 3.1 What “materially different” means

Material difference requires change in at least one of:

```text
algorithm family (e.g. serial reduce vs simdgroup reduce vs block tiled)
math domain (e.g. fast-math exp vs precise exp vs host-lib reconstruction)
command topology (per-dispatch CB vs fused multi-encoder one-CB)
parallel decomposition (sequence shard vs layer shard vs microbatch)
representation (bf16 authority vs fp8/fp4 candidate vs Q4_K)
```

**Not material:** learning rate of a search, thread count ±1 on same kernel,
re-running the same candidate with more warmups, renames.

Matches self-TG loop: “propose three materially different mechanisms” /
“not three knobs on one idea” (`ASCENSION_TG_GAUNTLET_HARNESS_PLAN.md`).

### 3.2 Bookkeeping types (ROADBLOCK pattern)

Tonight already proved the spirit of `ROADBLOCK_CANDIDATE` on the mHC path
(§4). Make it explicit:

```text
schema: hawking.ascension.optimization_attempt.v1
```

| Field | Type |
|-------|------|
| `attempt_id` | string |
| `model_id` / `family` | |
| `architecture_class` | string enum (see below) |
| `mechanism_class` | stable id for graveyard join |
| `hypothesis` | prose |
| `implementation_ref` | path / kernel name / worktree id |
| `parity_result` | `PASS_NUMERIC_V2_1_ONLY` / `REJECT_PARITY` / … |
| `wall_p50` / `wall_p99` | optional if parity failed first |
| `bottleneck_observed` | string id |
| `materially_new_premise` | bool — required true if retrying a buried mechanism |
| `seal_sha256` | |
| `status` | attempt terminal status |

```text
schema: hawking.ascension.roadblock_ledger.v1
```

| Field | Notes |
|-------|-------|
| `bottleneck_id` | stable |
| `mechanism_class` | |
| `attempts[]` | ordered optimization_attempt refs |
| `repeat_count` | attempts sharing mechanism_class + bottleneck |
| `architectures_failed` | distinct architecture_class set |
| `roofline_ref` | |
| `next_representation_change` | filled when B4 ready |
| `kind` | always report as `ROADBLOCK_CANDIDATE` until controller certifies |
| `graveyard_link` | optional bury id |

`architecture_class` starter enum (extend with evidence, don’t bikeshed):

```text
SERIAL_AUTHORITY
SIMDGROUP_REDUCE
BLOCK_TILED
SPLIT_K_PARALLEL
COMMAND_TOPOLOGY_FUSE
SEQUENCE_SHARD_HOST
LAYER_SHARD_RESIDUAL
FAST_MATH_TRANSCENDENTAL
PRECISE_MATH_TRANSCENDENTAL
HOST_LIB_RECONSTRUCT_ON_DEVICE
ULP_REPAIR_MICROKERNEL
REPRESENTATION_REQUANT
EXPERT_WAVE_CONCURRENCY
```

---

## 4. Worked example — mHC ULP / Darwin double-double (tonight)

This is the **canonical Trigger-B bookkeeping example**. It is **not** yet a
completed rotation of DeepSeek (optimization continues / family still active);
it shows how repeated mechanism + distinct architectures are recorded until a
general fix lands — and how that maps onto rule B’s fields.

### 4.1 Bottleneck sealed

| Field | Value |
|-------|-------|
| `bottleneck_id` | `mhc_control_exp_host_device_mismatch` |
| `mechanism_class` | `control_domain_exp_fidelity` |
| `symptom` | host (Darwin `expf` / Rust `f32::exp`) vs Metal exp disagree in mHC control domain; residual path cannot claim exact control / tight ULP |
| `geometry` | mHC control inputs finite in **[-40, 40]**; P4B one-thread 4-lane post+comb; Sinkhorn / sigmoid |

### 4.2 Materially different attempts (architecture_class)

Derived from real diagnostic binaries and kernels in-tree (read-only):

| # | architecture_class | Kernel / probe | What it tested | Outcome class |
|---|--------------------|----------------|----------------|---------------|
| 1 | `FAST_MATH_TRANSCENDENTAL` | `deepseek_v4_p4b_hc_control_fast_exp_trace_candidate` | default Metal fast-math `fast::exp` on control path | **fail** — host/device mismatch class persists |
| 2 | `PRECISE_MATH_TRANSCENDENTAL` | `deepseek_v4_p4b_hc_control_precise_exp_trace_candidate` / `…_hc_post_comb_precise_exp_candidate` | precise Metal exp / precise post+comb | **fail** — still not Darwin host bit identity |
| 3 | `ULP_REPAIR_MICROKERNEL` | `deepseek_v4_p4b_hc_post_cpu_exp_ulp_repair_trace_candidate` | trace-bound two-logit ULP repair after precise path | **fail as general fix** — diagnostic notes promotion prohibited; terminal BF16 untested; not a general domain solution |
| 4 | `HOST_LIB_RECONSTRUCT_ON_DEVICE` (FDLIBM port) | `deepseek_v4_p4b_fdlibm_expf_compat_domain_candidate` + host `fdlibm_expf_control_domain` | FreeBSD/FDLIBM expf on Metal | **fail vs Darwin target** — host target is Darwin libSystem, not FDLIBM |
| 5 | `HOST_LIB_RECONSTRUCT_ON_DEVICE` (Darwin DD) | `deepseek_v4_p4b_darwin_expf_dd_compat_domain_candidate` → promoted helper `deepseek_v4_mhc_control_exp.metal` (`darwin_double_double_control_domain_general`) | arm64 Darwin `expf` double-double table reconstruction on Metal (no device double) | **general fix for control-domain exp** — reseal path `DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v1-reseal-darwin-dd.json`; multi-layer receipts note `mhc_control_exp=darwin_double_double_control_domain_general` |

Host-side characterization (attempt 0, measurement only):

- `gravity_deepseek_v4_p4b_cpu_exp_compat_diagnostic.rs` — establishes **Rust `f32::exp` vs Darwin `expf`** as the true host target; disassembly reconstruction directions (`a_x_minus_n` / `n_minus_a_x`).

### 4.3 How this maps to Trigger B conjuncts

| Conjunct | mHC sequence |
|----------|--------------|
| B1 two architectures fail | At least attempts 1–4 are distinct classes that failed to clear the bottleneck before DD; even 1–3 alone already satisfy “≥2 materially different failures” |
| B2 roofline | P4B stage profile: mHC pre alone **~35–42 ms GPU** / dispatch on attention path; KERNEL_BROKERS notes attn pre **76.61 ms GPU / 2 dispatches** in P4B profile (75% of attention GPU) — control path is a named cost center |
| B3 repeated bottleneck sealed | Same `control_domain_exp_fidelity` / host-vs-Metal exp mismatch across fast, precise, ULP-repair, and FDLIBM attempts |
| B4 smallest next representation change | Named and taken: **reconstruct Darwin expf as software float double-double on Metal** for the control domain — not “bigger TG”, not “ignore ULP” |

### 4.4 Why this is **not** an automatic family rotation

Rule B is for when optimization is **exhausted** and the next step is a
**representation / subject change** or model rotate. Here the DD fix **unblocked**
the family; the correct ledger outcome is:

```text
ROADBLOCK_CANDIDATE → PROMOTED_MECHANISM (controller)
mechanism retained as FAMILY_SPECIFIC grammar (mHC)
process retained as CROSS_FAMILY (transcendental fidelity discipline)
continue OPTIMIZE on same model
```

If instead DD had failed and the next named change were e.g. “drop exact mHC
control and accept looser residual only” or “move to a non-mHC family,” B4 would
point at that representation change and rotation/family switch could proceed.

### 4.5 Freeze-by-geometry (post-fix)

Winning grammar to freeze before any DeepSeek source eviction:

| Geometry key | Frozen artifact |
|--------------|-----------------|
| `family=DEEPSEEK_V4`, `operator_grammar=mhc_control_exp`, `representation=darwin_dd_table_v1`, `context_regime=control_domain_[-40,40]` | `deepseek_v4_mhc_control_exp.metal` + reseal seal_sha256 |
| P4A one-CB attention topology | `DSV4F_P4A_LAYER0_ATTENTION_TOPOLOGY_SWEEP-v1.json` (`promoted: true`) |
| Sequence-shard capture | `FULLSEQ_CAPTURE_PARALLELISM_FINDINGS.json` + merge tool contract |

Parity honesty after DD: multi-layer paths still
`NUMERIC_PARITY_V2_1_ONLY` until exact-storage e2e — freeze does **not** inflate
claim_boundary.

---

## 5. Rotation decision receipt

```text
schema: hawking.ascension.rotation_decision.v1
```

| Field | Notes |
|-------|-------|
| `trigger` | `TG_RUNG_DESCENT` \| `TWO_FAILED_ARCHITECTURES` \| `TG3_HUMAN_REVIEW` |
| `trigger_receipt_seal` | A or B receipt |
| `model_from` / `family_from` | |
| `model_to` / `family_to` | optional; may be null if stop |
| `frozen_kernel_grammars[]` | selection keys + seal_sha256 — **required non-empty before EVICT** |
| `evict_authorized` | bool; default false until freeze verified |
| `status` | `ROTATE_CANDIDATE` / `ROTATE_CERTIFIED` / `HOLD_TG3` / `REJECT_ROTATION` |
| `principal` | sandbox may propose `ROTATE_CANDIDATE` only |
| `claim_boundary` | |

### 5.1 Ordering with model ladder

```text
… → OPTIMIZE → REVIEW → REPORT → SEAL (freeze grammars)
→ (optional EVICT if authorized)
→ ROTATE (this plan)
→ DISCOVER (next) or stop
```

Illegal:

```text
EVICT without freeze
ROTATE on size alone
ROTATE past TG3 without human
declare FAMILY_EXHAUSTED from sandbox
```

---

## 6. Interaction with Graveyard (§32)

Before any new attempt:

```text
check_proposal(mechanism_class) → buried?
  if buried and not materially_new_premise → refuse
```

After Trigger B:

```text
bury failed architectures with:
  mechanism, model/geometry, measured outcome, failure reason, reopen condition
```

Known classes that often appear in rotation bookkeeping:

```text
fewer waits but slower wall time
unmeasured GPU claims
circular parity oracle
synthetic activation mismatch
prompt-independent collapse
```

mHC lesson for graveyard entries: “Metal `exp` ≈ host `exp`” without naming
**which host lib** is a non-reopenable vague burial; reopen requires a new
premise (e.g. new device libSystem, new control domain).

---

## 7. Interaction with family kernel transfer

On rotate:

1. Write transfer ledger entries for any win earned on the departing model.
2. Tag `transfer_class` (see family kernel plan).
3. Next model’s `RESEARCH_DISTINCTION` must cite transferable grammars
   (sequence shard, CB fuse, …) and must **not** assume family-specific math
   (mHC DD) applies.

---

## 8. Implementation sequence (scaffold → live)

| Step | Deliverable |
|------|-------------|
| 0 | Types: `optimization_attempt`, `roadblock_ledger`, `tg_rung_descent_receipt`, `rotation_decision` |
| 1 | Unit tests: material difference predicate; B requires all four conjuncts; TG3 forces `HOLD_TG3` |
| 2 | Encode mHC sequence as golden fixture for roadblock ledger (read-only paths to kernels/receipts) |
| 3 | Wire self-TG loop to append `optimization_attempt` rows automatically |
| 4 | Controller certification path only for `ROTATE_CERTIFIED` / serve promote |
| 5 | Eviction tool refuses without `frozen_kernel_grammars[]` |

---

## 9. Success criteria for §31

```text
[ ] Trigger A emits sealed descent receipt with BASE_TRUE_TPS only
[ ] Trigger B refuses if any conjunct missing
[ ] TG3 always yields TG3_REVIEW_REQUIRED / HOLD_TG3
[ ] mHC worked example loads as roadblock ledger fixture
[ ] EVICT blocked without freeze-by-geometry list
[ ] Sandbox cannot certify ROTATE or FAMILY_EXHAUSTED
```

---

## 10. Honesty / non-claims

- mHC DD is a **mechanism promotion**, not proof that DeepSeek has finished its
  TG ladder or should rotate away.
- P4A one-CB and sequence-shard wins do not by themselves satisfy Trigger A or B.
- Component SIMDgroup speedups without full residual + capability are not TG
  descents.
- This plan does not authorize deletion of Gravity streams or HF caches.
