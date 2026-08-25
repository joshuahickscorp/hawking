# Ascension Wider Autonomous Model Ladder Plan

**Status:** PLAN ONLY — future programme, gated on Proto-Frankenstein offload.  
**Bible:** HAWKING_ASCENSION_BIBLE §30 (Wider autonomous model ladder), with §7–§9, §29, §31.  
**Companion plans:** `ASCENSION_FAMILY_KERNEL_ARCHITECTURE_PLAN.md`, `ASCENSION_ROTATION_RULE_PLAN.md`  
**Scaffold seeds:**

- `lab/operators/ascension_parity_ladder.py` — `MODEL_LADDER_PIPELINE`, `model_ladder_pipeline_receipt()`
- `lab/operators/research_registry.py` — `ADMIT_TO_*` / `DEFER` / `REJECT`
- `lab/verification_authority.py` — candidate vs authoritative claims
- `workspace/ops/ascension/notifications.py` — `NEW_MODEL_ADMITTED`, `PARITY_REJECTION`
- Tonight’s DSV4F stage vocabulary and sealed receipts (read-only)

**Explicit non-goals:** no Qwen/Gravity downloads, no model streaming, no push/PR,
no detached daemons, no committing a venv.

---

## 1. Purpose

After Qwen bootstrap (and with DSV4F as the living exact-model reference), the
runtime must climb a **wider autonomous model ladder**:

```text
DISCOVER
→ PREFLIGHT
→ RESEARCH_DISTINCTION
→ DOWNLOAD_STREAM
→ GRAVITY
→ LOAD
→ PARITY
→ CAPABILITY
→ PROFILE
→ OPTIMIZE
→ REVIEW
→ REPORT
→ SEAL
→ EVICT
→ ROTATE
```

Select models for **new physical or architectural distinction**, not size.

Every admitted model answers:

```text
What new physical or architectural distinction does this test?
Which existing kernel grammar should transfer?
Which new operator or state contract is required?
What evidence would make the model redundant?
```

---

## 2. Design principle: reuse tonight’s vocabulary

Do **not** invent a second status system. Generalize DSV4F / research / parity
terms already in-tree.

### 2.1 Stage names (pipeline cursor)

Exact bible order, underscored for types (already in scaffold):

| Stage | Meaning |
|-------|---------|
| `DISCOVER` | Candidate model identity + public architecture signals |
| `PREFLIGHT` | Storage, credential, pressure, Proto-Frankenstein gate |
| `RESEARCH_DISTINCTION` | Four questions + research registry items |
| `DOWNLOAD_STREAM` | Pin revision, stream under floor policy, seal SHA |
| `GRAVITY` | Gravity co-design ladder (source → equilibrium) |
| `LOAD` | Artifact load / residency mode |
| `PARITY` | P0–P13 ladder; `NumericParityV21Only` until exact-storage |
| `CAPABILITY` | Tool/JSON/edit / product tests |
| `PROFILE` | Complete-token profiler + FLOPS ledger |
| `OPTIMIZE` | Self-TG loop; broker A/B; graveyard-checked |
| `REVIEW` | Adversarial / controller review; TG3 hard stop |
| `REPORT` | Candidate report only (`hawking.lab.candidate_report.v1`) |
| `SEAL` | Sealed receipts; freeze kernel grammar by geometry |
| `EVICT` | Source eviction only after freeze + hash-verify |
| `ROTATE` | Apply rotation rule A or B (see rotation plan) |

### 2.2 Terminal statuses (no synonyms)

Reuse `RungStatus` / research verdicts:

| Class | Values |
|-------|--------|
| Scaffold honesty | `SCAFFOLD_PENDING`, `REJECT_WEIGHTS_ABSENT`, `REJECT_ARTIFACT_ABSENT` |
| Execution honesty | `REJECT_FALLBACK_NONZERO`, `REJECT_NO_REAL_GPU_DISPATCH`, `REJECT_PARITY`, `REJECT_CAPABILITY`, `REJECT_CLAIM_BOUNDARY` |
| Partial pass | `PASS_DIAGNOSTIC_ONLY`, `PASS_NUMERIC_V2_1_ONLY`, `PASS_PARTIAL`, `PASS_SCAFFOLD_CONTRACT` |
| Full | `PASS_FULL_STACK` |
| Metric honesty | `BASE_TRUE_TPS_WITHHELD`, `METRIC_WITHHELD` |
| Human gates | `TG3_REVIEW_REQUIRED`, `HUMAN_REVIEW_REQUIRED` |
| Research admit | `ADMIT_TO_GRAVITY`, `ADMIT_TO_RUNTIME`, `ADMIT_TO_KERNEL`, `DEFER`, `REJECT` |
| DSV4F-style seals | e.g. `PASS_REAL_METAL_*_NOT_RUNTIME`, `PASS_MULTI_LAYER_GPU_FORWARD_*` — **claim-bounded**, not full stack |

### 2.3 Parity classification

```text
SCAFFOLD_PENDING
NUMERIC_PARITY_V2_1_ONLY    # mirrors NumericParityV21Only
EXACT_STORAGE
REJECTED
```

### 2.4 Candidate kinds (sandbox may emit)

From `CandidateKind` — include tonight’s roadblock pattern:

```text
candidate_mechanism
implementation_receipt
parity_evidence
capability_evidence
benchmark_evidence
review_objection
known_limitation
recommended_next_experiment
repetition_fingerprint
ROADBLOCK_CANDIDATE
TG_RUNG_CANDIDATE
```

Sandbox models **must not** emit `PROMOTED` / `COMPLETE` / `PHYSICAL_LIMIT_REACHED` /
`FAMILY_EXHAUSTED` / `SAFE_TO_DELETE` / `FINAL_VERDICT`.

---

## 3. Selection for distinction (not size)

### 3.1 Distinction catalogue (bible §30)

A model is interesting only if it adds at least one of:

| Distinction code | Examples |
|------------------|----------|
| `EXPERT_GEOMETRY` | top-k, n_experts, shared expert presence |
| `ROUTING` | hash route, learned bias, softmax gate, no router |
| `ATTENTION_OR_STATE` | GQA/MHA, sparse sink, DeltaNet, SSM, mHC |
| `QUANT_LAYOUT` | FP4/FP8 native, Q4_K/Q6_K, UE8M0 act quant |
| `ACTIVE_WORKING_SET` | expert residency pressure, stream vs resident |
| `CONTEXT_BEHAVIOR` | 128K/256K, growing KV, ratio-4/128 sparse |

**Reject as sole reason:** “larger than previous.”

### 3.2 Admission dossier (required)

```text
schema: hawking.ascension.model_distinction_dossier.v1
```

| Field | Type | Required answer |
|-------|------|-----------------|
| `model_id` | string | HF id or sealed locator |
| `family` | `ModelFamily` | one of six |
| `distinctions[]` | enum list | from catalogue |
| `new_physical_or_architectural_distinction` | prose + codes | Q1 |
| `kernel_grammar_transfer[]` | transfer ledger ids | Q2 — cite DSV4F wins where applicable |
| `new_operator_or_state_contract` | string list | Q3 |
| `redundancy_evidence` | prose | Q4 — what would make this model unnecessary |
| `research_verdict` | ADMIT_* / DEFER / REJECT | from registry |
| `claim_boundary` | object | default_claim_boundary |
| `status` | RungStatus-like | usually `PASS_SCAFFOLD_CONTRACT` or `REJECT_*` |

### 3.3 Worked mapping — DSV4F as living `DEEPSEEK_V4` admission

| Question | Tonight’s answer (evidence-backed) |
|----------|-------------------------------------|
| New distinction? | mHC control path; sparse attention ratios; source-native FP4 experts + FP8 control; top-6 hash then learned-bias route; 43-layer streamed Gravity body |
| Kernel grammar that transfers? | CB collapse; sequence sharding for host-serial capture; MoE batch/wave structure; act-quant + simdgroup candidate ladders; NumericParity V2.1 process |
| New operator/state contract? | mHC HC state + Darwin DD exp; P6 two-phase expert load; growing-KV sparse sink; full_stream Gravity schema |
| Redundancy evidence? | Another model with identical mHC+FP4+sparse contracts and no new quant/context pressure would be `REJECT` as redundant after transfer ledger saturation |

Initial six families already encode the intended distinction axes; the ladder
fills **instances**, not infinite near-duplicates.

---

## 4. Pipeline stage design (types + receipts)

### 4.1 Cursor receipt

```text
schema: hawking.ascension.model_ladder_pipeline.v1   # already scaffolded
```

| Field | Notes |
|-------|-------|
| `family` | `ModelFamily` |
| `model_id` / `revision_pin` | filled at DOWNLOAD_STREAM |
| `pipeline` | full ordered list |
| `current_phase` | one of the 15 |
| `phase_history[]` | `{phase, entered_at, status, receipt_seal}` |
| `rotation_triggers` | enum list from rotation plan |
| `selection_questions` | the four bible questions |
| `claim_boundary` | always present |
| `status` | aggregate honesty |

### 4.2 Per-phase gate table

| Phase | Entry require | Exit status examples | Primary receipt schema |
|-------|---------------|----------------------|------------------------|
| DISCOVER | none | dossier draft | `model_distinction_dossier.v1` |
| PREFLIGHT | pressure governor OK; Proto-Frankenstein offload for Qwen path | `PASS_SCAFFOLD_CONTRACT` or `REJECT_*` | `state_gate` / pressure receipt |
| RESEARCH_DISTINCTION | dossier four questions complete | research items all decided for Gravity-affecting mechanisms | `research_registry` items |
| DOWNLOAD_STREAM | ADMIT from research; credentials; floor free | stream seal + SHA | acquisition / stream seal |
| GRAVITY | streamed weights | gravity ladder stages | `gravity_ladder_receipt.v1` |
| LOAD | Gravity artifact | residency mode A/B/C | load receipt |
| PARITY | load OK | P0–P13; often `PASS_NUMERIC_V2_1_ONLY` long before full | `parity_ladder_receipt.v1` |
| CAPABILITY | parity floor for capability rungs | tool/JSON product tests | HCLI product catalog |
| PROFILE | capability floor | complete-token profile ≥98% wall named | profiler receipt |
| OPTIMIZE | profiled bottlenecks | TG candidates; broker A/B | TG + kernel selection receipts |
| REVIEW | optimize reports | `TG3_REVIEW_REQUIRED` at TG3 | authoritative verdict (controller) |
| REPORT | always continuous | candidate_report only | `hawking.lab.candidate_report.v1` |
| SEAL | review pass or bounded diagnostic seal | seal_sha256; freeze grammar by geometry | seal receipts |
| EVICT | freeze complete; hash-verify | source removed from active envelope | eviction receipt |
| ROTATE | rule A or B satisfied | next model DISCOVER or stop | rotation decision receipt |

### 4.3 Phase transition rules

```text
cannot skip RESEARCH_DISTINCTION before DOWNLOAD_STREAM
cannot claim PASS_FULL_STACK from PARITY alone
cannot OPTIMIZE without PROFILE bottleneck rank
cannot EVICT without SEAL freeze of winning kernel grammar by geometry
cannot ROTATE on size alone
TG3 → always REVIEW with TG3_REVIEW_REQUIRED (no autonomous promote)
```

---

## 5. Gravity sub-ladder (inside GRAVITY phase)

Reuse bible §8 / scaffold — **not** a universal 1.5 BPW mandate:

```text
source_authority
→ quality_anchor
→ performance_anchor
→ gravity_equilibrium_artifact
```

Target: **lowest capable, runnable equilibrium**.  
`universal_1_5_bpw_required: false` always.

Research admit mapping (from `ASCENSION_RESEARCH_REGISTRY_PLAN.md`):

| Research verdict | Prefer |
|------------------|--------|
| layout-affecting | `ADMIT_TO_GRAVITY` |
| kernel grammar | `ADMIT_TO_KERNEL` |
| runtime adapter | `ADMIT_TO_RUNTIME` |
| later | `DEFER` |
| ruled out | `REJECT` |

---

## 6. Parity sub-ladder (inside PARITY phase)

P0–P13 shared skeleton (`PARITY_RUNGS`); family stages differ
(`FAMILY_STAGES`).

Promotion rules (DSV4F, already coded in `promote_rung_status`):

```text
fallback_count != 0     → REJECT_FALLBACK_NONZERO
gpu_dispatches == 0     → REJECT_NO_REAL_GPU_DISPATCH
parity fail             → REJECT_PARITY
numeric only            → PASS_NUMERIC_V2_1_ONLY
full residual+capability+fallback=0+real GPU → PASS_FULL_STACK
no weights              → REJECT_WEIGHTS_ABSENT / SCAFFOLD_PENDING
```

DSV4F multi-layer precedent: L0–L1/L2/L42 receipts classify
`NUMERIC_PARITY_V2_1_ONLY` with explicit reason that exact-storage e2e is not
yet earned — the ladder must keep that honesty when generalizing.

---

## 7. Transfer-aware DOWNLOAD / OPTIMIZE

When a new model enters `RESEARCH_DISTINCTION` / `OPTIMIZE`:

1. Load `KernelGrammarTransferLedger` (family kernel plan).
2. For each CROSS_FAMILY_TOPOLOGY entry (sequence shard, CB fuse, physical
   trace, act-quant grammar), mark **try-first**.
3. For FAMILY_GRAMMAR entries, try only if `family` matches or geometry adapter
   exists.
4. For FAMILY_SPECIFIC (mHC DD math, DSV4 FP4 packing), **do not** port; only
   port the *process* (roadblock bookkeeping, control-domain fidelity tests).
5. Record outcomes as new ledger entries (promote/demote transfer_class with
   evidence).

This is how DSV4F pays forward into Qwen / Llama / Mixtral without cargo-culting
DeepSeek tensors.

---

## 8. Notifications & authority

| Event | NotificationKind (existing) | Principal |
|-------|----------------------------|-----------|
| Model passes PREFLIGHT+RESEARCH admit | `NEW_MODEL_ADMITTED` (candidate until controller certifies) | controller certifies |
| Parity reject | `PARITY_REJECTION` | automatic record |
| TG3 | `TG3_REVIEW_REQUIRED` status + notify human | human/controller |
| Roadblock | `ROADBLOCK_CANDIDATE` kind | sandbox propose; controller may seal |

---

## 9. Ordering relative to bootstrap

Governing order (bible §0 compressed):

```text
Proto-Frankenstein offload
→ research + Gravity co-design for Qwen
→ exact-model 30B (QWEN3_MOE)
→ self-TG → TG3 review
→ exact-model 80B hybrid (QWEN3_NEXT)
→ self-TG → TG3 review
→ promote executor+reviewer → Agent OS → Option-C
→ generalize family megakernels (§29)
→ wider ladder rotation (§30–§31)
```

**DSV4F tonight** sits as the **methodology and transfer-ledger donor** inside
family `DEEPSEEK_V4`. It does not replace Qwen bootstrap as HCLI workforce.

Current scaffold cursor for Qwen families: `current_phase: PREFLIGHT`.

---

## 10. Type inventory

| Schema | Owner |
|--------|-------|
| `hawking.ascension.model_ladder_pipeline.v1` | cursor |
| `hawking.ascension.model_distinction_dossier.v1` | four questions |
| `hawking.ascension.model_ladder_phase_receipt.v1` | one phase enter/exit |
| `hawking.ascension.parity_ladder_receipt.v1` | PARITY |
| `hawking.ascension.gravity_ladder_receipt.v1` | GRAVITY |
| `hawking.lab.candidate_report.v1` | REPORT |
| `hawking.lab.authoritative_verdict.v1` | controller REVIEW |
| research registry schemas | RESEARCH_DISTINCTION |

---

## 11. Success criteria for §30

```text
[ ] Pipeline cursor advances only via sealed phase receipts
[ ] No model admitted without four distinction answers
[ ] No DOWNLOAD_STREAM without research decisions on Gravity-affecting mechanisms
[ ] Transfer ledger consulted at RESEARCH_DISTINCTION and OPTIMIZE
[ ] PASS_FULL_STACK never asserted from partial DSV4F-style claim boundaries
[ ] Size-only candidates rejected at DISCOVER/RESEARCH with explicit reason
```

---

## 12. Honesty / non-claims

- Ladder automation does not grant sandbox models authority.
- DSV4F multi-layer `PASS_MULTI_LAYER_GPU_FORWARD_*` is **not** `PASS_FULL_STACK`.
- Sequence-sharding and SIMDgroup component wins do not admit a new HF model by
  themselves.
- This plan does not schedule concurrent multi-family downloads against the
  pressure governor.
