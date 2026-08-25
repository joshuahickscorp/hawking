# Residual Teacher Admission Gate

**Status:** local evidence gate + scaffold  
**Scaffold:** `lab/operators/residual_teacher_admission_gate.py`  
**Tests:** `lab/tests/test_residual_teacher_admission_gate.py`  
**Related:** Stage-1 GLMâDSV4F Proto; Stage-2 Kimi plan (`STAGE2_KIMI_STREAMING_DISTILL_PLAN.md`)

---

## 1. Decision

The second teacher (Kimi) is **not** an assumed duplicate of the GLM transfer.

After GLM-to-DSV4F Proto, Kimi may enter only as an **evidence-gated residual lane**:

| Default | Meaning |
|---------|---------|
| **DEFERRED** | Missing or incomplete evidence; do nothing |
| **REJECT** | Hard requirement failed (mismatch, regression, no gain, forward closed, â¦) |
| **ADMIT** | Residual-lane permission only â not causal inheritance proof |

`ADMIT` does **not** mean âFinal Frankenstein has Kimi,â does **not** re-run a full dual-donor transfer, and does **not** claim that measured deltas are causal.

---

## 2. Why residual, not duplicate

Stage-1 already spends the Proto budget on GLM mathematical / latent transfer into the DeepSeek-V4-Flash body. A second full-scale teacher pass that only says âmore distillationâ is:

- storage-hostile (GLM and Kimi must not both stay resident),
- scientifically weak (no named residual capability),
- easy to fake as progress without held-out membership control.

Kimi is justified only when it targets a **named residual hypothesis** that Proto did not already cover (e.g. long-horizon agentic planning, multi-tool orchestration), and only when incremental held-out evidence is positive on the **same** evaluation membership as the sealed GLM-only baseline.

---

## 3. Admission requirements (all required)

The gate validates supplied JSON-like evidence only â no networking, Hub/Xet, model load, cache write, download, subprocess, or trainer launch.

1. **Sealed GLM-only baseline receipt**  
   Proto / GLM baseline sealed; teacher set is GLM-only (Kimi not smuggled into baseline).

2. **Kimi incremental A/B receipt**  
   Same `held_out_membership_hash` as the baseline; structure identifies a Kimi treatment arm.

3. **Positive incremental held-out improvement**  
   Explicit delta or A/B scores with treatment > baseline. Zero or negative â **REJECT**. Absent â **REJECT** when a Kimi claim is present.

4. **Named residual hypothesis / capability**  
   Not generic âmore distillationâ / âadd Kimiâ language.

5. **Provenance / revision identity**  
   Bound student (DSV4F), GLM baseline, and Kimi revision identities.

6. **Explicit no-regression** on protected axes: **math, coding, tool, agentic**.

7. **DSV4F architecture / forward gate ready**  
   If forward is `DEEPSEEK_FORWARD_PENDING` / not ready â **REJECT**.

---

## 4. Reject conditions (hard)

- Incremental held-out improvement absent or non-positive  
- Protected axis regression (math / coding / tool / agentic)  
- Evaluation membership hash differs from GLM baseline  
- DSV4F architecture/forward gate not ready  
- Unsealed or non-GLM-only baseline  
- Generic distillation hypothesis  
- Incomplete provenance when a claim is made

Incomplete *missing* evidence (no Kimi claim yet) stays **DEFERRED**, not a fabricated **ADMIT**.

---

## 5. Claim boundary

```text
ADMIT  â  may start residual Kimi lane under this evidence bundle
       â  causal proof that Kimi caused the delta
       â  permission to hold GLM + Kimi resident
       â  Final Frankenstein / Stage-2 stream launch by itself
       â  duplicate full GLM-style transfer assumed complete
```

Stage-2 storage, stream launch, and owner/runtime gates remain separate (see Stage-2 plan). This document only freezes the **admission decision** shape.

---

## 6. Operator API

```python
from lab.operators.residual_teacher_admission_gate import (
    evaluate_residual_teacher_admission,
    default_decision,
)

decision = default_decision()          # always DEFERRED
decision = evaluate_residual_teacher_admission(evidence_dict)
# or keyword fragments: glm_baseline_receipt=..., kimi_incremental_receipt=..., ...
```

Returned document schema: `hawking.residual_teacher.admission_gate.v1`  
Fields: `verdict`, `reason`, `reasons[]`, `checks[]`, `requirements`, `claim_boundary`, `local_only`.

---

## 7. Relation to programme

| Stage | Role of this gate |
|-------|-------------------|
| Proto (GLMâDSV4F) | Must produce the sealed GLM-only baseline this gate consumes |
| Residual Kimi lane | Opens only on **ADMIT** |
| Stage-2 streaming plan | Still owns storage floors, single-donor windows, stream launch |
| Odyssey / Ramanujan | Unrelated authority; not granted by this gate |
