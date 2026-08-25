# Ascension Family Kernel / Model Ladder / Rotation Contract (Bible §§29–31)

**Status:** CONTRACT SEALED (planning only) — implementation **NOT_STARTED**  
**Bible:** `HAWKING_ASCENSION_BIBLE.md` §29 (Family kernel), §30 (Wider model ladder), §31 (Rotation)  
**Machine-readable registry:** [`ASCENSION_BLOCKED_STATE_REGISTRY.json`](./ASCENSION_BLOCKED_STATE_REGISTRY.json)  
**Scaffold code:** `lab/operators/ascension_parity_ladder.py`, `lab/operators/ascension_contracts.py`  
**Related:** 30B/80B parity plans, TG harness, profiler contract  
**Schedule steps:** 28 (family graphs), 29 (wider ladder), 30 (rotation) — all `NOT_STARTED`

---

## 1. Purpose

Provide one **canonical machine-readable state registry** for bootstrap models
and family-ladder work so Qwen identities cannot be treated as ready, admitted
to sandbox workforce, or generalized into megakernels without sealed gates.

This contract **does not** mark any model, schedule step, or completion state
ready. Both bootstrap models start **`BLOCKED`**.

---

## 2. Bootstrap models — BLOCKED by default

| Display name | Family | Role | Registry id | Status |
|--------------|--------|------|-------------|--------|
| **Qwen3-Coder-30B** | `QWEN3_MOE` | executor | `qwen3_coder_30b` | **BLOCKED** |
| **Qwen3-Coder-Next-80B** | `QWEN3_NEXT` | reviewer | `qwen3_coder_next_80b` | **BLOCKED** |

HF placeholders (pin revision at stream time):

```text
Qwen/Qwen3-Coder-30B-A3B-Instruct
Qwen/Qwen3-Coder-Next
```

Neither model is sandbox-admitted until **all** required blockers clear.

---

## 3. Required blocker fields

Every registry entry **must** carry these six blockers (all start `BLOCKED`):

| Field | Cleared only when |
|-------|-------------------|
| `architecture_config_source_admission` | Sealed config/source admission + revision pin |
| `gravity_family_support` | Sealed Gravity family-support receipt for that identity |
| `exact_runtime` | Sealed exact-model runtime + required parity rungs |
| `profiler_evidence` | Sealed complete-token profile (≥98% explained) + FLOPS ledgers (bible §11) |
| `tg_evidence` | Sealed Self-TG gauntlet under separated metrics |
| `tg3_approval` | Human/controller certification of TG3 review (no self-promote) |

Clearing rules:

1. Evidence refs must be non-empty sealed receipt paths or authority ids.
2. Sandbox may propose a clear; only **controller/human** may certify.
3. Clearing one model does **not** clear the other.
4. Family megakernel promotion (§29) requires exact-model success first on that family.

---

## 4. Family kernel architecture (bible §29)

After exact-model success, generalize into:

```text
shared semantic runtime
architecture-family execution graphs
generated geometry variants
exact-model fast paths where justified
```

Initial families:

```text
QWEN3_MOE
QWEN3_NEXT
DEEPSEEK_V4
LLAMA
MISTRAL_MIXTRAL
STATE_SPACE_HYBRID
```

Kernel selection key:

```text
family semantics
operator grammar
tensor dimensions
representation
active experts
batch/session count
context regime
device generation
memory pressure
```

A megakernel may be one Metal function **or** one fused replayable command graph.

**Promote only by complete wall time and p99** — not microbench alone, not FLOPS alone.

---

## 5. Wider autonomous model ladder (bible §30)

Pipeline (must match registry + parity scaffold):

```text
DISCOVER → PREFLIGHT → RESEARCH_DISTINCTION → DOWNLOAD_STREAM → GRAVITY
→ LOAD → PARITY → CAPABILITY → PROFILE → OPTIMIZE → REVIEW → REPORT
→ SEAL → EVICT → ROTATE
```

Current phase for both bootstrap entries: **`PREFLIGHT`** (scaffold), still
under outer status **`BLOCKED`**.

Select models because they add new geometry / routing / state / quant layout /
working-set pressure / context behavior — **not solely because they are larger**.

Every admitted model answers:

```text
What new physical or architectural distinction does this test?
Which existing kernel grammar should transfer?
Which new operator or state contract is required?
What evidence would make the model redundant?
```

Completion state `HAWKING_SELF_CONTAINED_MODEL_LADDER_ACTIVE` remains
**CANDIDATE** until schedule step 29 is controller-certified.

---

## 6. Rotation rule (bible §31)

Rotate when either:

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

At TG3:

```text
always stop for human/controller review
```

Freeze every winning kernel grammar by geometry **before** source eviction.

---

## 7. Integration (without marking stages ready)

| Tracker | How this contract plugs in |
|---------|----------------------------|
| Overview §29–31 rows | Point at this contract + blocked registry (not “pending companion” for the contract files) |
| Master schedule steps 28–30 | Companion docs include this contract + registry |
| Steps 11–16 / 17–23 | Bootstrap work remains gated by BLOCKED entries |
| Profiler contract | Supplies the `profiler_evidence` bar |
| Platform contract | Family graphs remain Metal Tier-1; portable IR only where listed |
| Completion states | All related states stay `CANDIDATE` |

**Landing this registry does not advance** any schedule step status.

---

## 8. Honesty

```text
any_bootstrap_model_unblocked = false
family_megakernels_promoted   = false
wider_ladder_active           = false
stage_ready                   = false
```

No live Qwen weights, runtime, profiler values, TG rungs, or sandbox workforce
admission are claimed by this document.

---

## 9. Acceptance for this planning contract

- [x] Registry names Qwen3-Coder-30B and Qwen3-Coder-Next-80B as **BLOCKED**.
- [x] Six required blocker fields present and defined on every entry.
- [x] Family list, ladder pipeline, rotation A/B + TG3 freeze rule recorded.
- [x] Machine-readable JSON + structural validation tests.
- [ ] First blocker field cleared with sealed evidence (post-gate).
- [ ] Either bootstrap model unblocked after TG3 approval.
- [ ] Family graphs / wider ladder under certified steps 28–29.

---

## 10. Non-goals

- No Qwen download, Gravity pack, or megakernel implementation in this wave.
- No self-promotion past TG3.
- No silent flip of `BLOCKED` → ready without evidence + certification.
- No schedule / completion-state status flips merely to make docs look complete.
