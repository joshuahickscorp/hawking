# Frankenstein Programme — Full Multi-Stage Plan

**Status:** PLAN ONLY (authoritative programme doc for the 3-generation path).  
**Seal companion:** `workspace/campaign/evidence/models/frankenstein/FRANKENSTEIN_PROGRAM_PLAN.json`  
**Related plans:**  
- Stage-2 detail → `STAGE2_KIMI_STREAMING_DISTILL_PLAN.md`  
- Memory / BPW → `RESIDENT_MEMORY_AND_BPW_PLAN.md`  
- Post-Final / HCLI → `POST_FINAL_GRAVITY_HCLI_PLAN.md`

---

## 1. Naming freeze (do not drift)

| Frozen name | Meaning |
|-------------|---------|
| **DeepSeek-V4-Flash Gravity** | Plain Terra-role model — DeepSeek-V4-Flash body under Gravity packaging, **no** Frankenstein donor inheritance |
| **Proto-Frankenstein** | DeepSeek + GLM mathematical inheritance |
| **Final Frankenstein** | DeepSeek + GLM + Kimi strategic/agentic inheritance |
| **Ramanujan** | Final Frankenstein **after** Odyssey / verified-training / Q-gauntlet / sandbox qualification |
| **Qwen Gravity** | Luna-role executor — `Qwen3-Coder-30B-A3B` (30.5B total / 3.3B active) under Gravity |

**Role map (HCLI tiers):**

| Role | Model line | Job |
|------|------------|-----|
| **Sol** (flagship) | Final Frankenstein → Ramanujan | Long-horizon strategy, hard reasoning, orchestration of other tiers |
| **Terra** | DeepSeek-V4-Flash Gravity (plain) | Strong general / agent body without donor fusion |
| **Luna** | Qwen Gravity | Fast coding executor, tool loops, cheap iteration |

Sol is **flagship positioning**, not "delete coding." Coding, tool-use, and
general reasoning remain first-class; Odyssey **adds** formal/verified
strength without erasing them (additive law below).

---

## 2. Three stages (settled architecture — detail, do not redesign)

```
Stage 1   GLM math  →  DeepSeek-V4-Flash body   =  PROTO_FRANKENSTEIN     (active)
Stage 2   Kimi K3 strategic  →  Proto             =  FINAL_FRANKENSTEIN
Stage 3   Odyssey + formal tools + verifier
          + Q-gauntlet + sandbox qualification    =  RAMANUJAN
```

### Stage 1 — Proto (active)

- Student body: DeepSeek-V4-Flash (284B total / 13B active, 43L, H=4096, MoE 256/top-6).  
- Donor: GLM-5.2 mathematical director (H=6144, 78L) via `GLM_MATH_BRIDGE`.  
- Mechanism: block-wise streaming distillation via latent bridge; residual
  adapters; **no** direct weight transplant.  
- Storage: one donor window at a time; raw DeepSeek body retained read-only
  until a verified successor exists.

### Stage 2 — Final

- Prerequisite: Proto shards sealed; **clear storage** of GLM windows.  
- Donor: Kimi K3 (2.8T / 104B active, 93L, H=7168, MoE 896/top-16, 1M ctx,
  agentic + coding + multimodal).  
- Bridge: `KIMI_STRATEGIC_BRIDGE`.  
- Same bounded-evicting stream as GLM; double-stream sequential density
  (see Stage-2 plan).  
- Output name: **Final Frankenstein**.

### Stage 3 — Ramanujan

- Input: Final Frankenstein (not a re-merge of raw teachers).  
- Adds:
  - **Odyssey** verified-training / research loop (T/F/Q programmes in
    `ramanujan/`).
  - **Formal tools** (Lean/Mathlib clean container and lattice of verifiers).
  - **Verifier dispositions** as promotion authority (roles economics:
    generators do not self-promote).
  - **Q-gauntlet** (Q0–Q6 contracts; Q0 clean-replay closed as historical
    container evidence; later Qs owner-gated).
  - **Sandbox qualification** (capability under restricted execution, not
    vibes).
- Output name: **Ramanujan**.  
- Explicit law: Odyssey **turns Final into Ramanujan without erasing**
  coding / agentic / tool-use / general reasoning. Those axes are
  **preserved and re-gated**, not sacrificed for formal score.

---

## 3. The four-way comparison (A/B/C/D)

Required contribution proof. **Do not claim donor value without this ablation.**

| ID | Name | Composition | What it answers |
|----|------|-------------|-----------------|
| **A = BASE** | DeepSeek-V4-Flash Gravity | Student body only; no Frankenstein adapters | Floor capability, TPS, resident memory, HCLI baseline |
| **B = PROTO** | Proto-Frankenstein | A + GLM math inheritance | Does math inheritance add without wrecking base? |
| **C = FINAL** | Final Frankenstein | B + Kimi strategic inheritance | Does strategic/agentic inheritance add without erasing math or base? |
| **D = RAMANUJAN** | Ramanujan | C after Odyssey + verifier + Q-gauntlet + sandbox | Does verified-training raise formal/reliability axes without capability regression on coding/agentic/general? |

### Evaluation contract (same for A–D)

- Same machine: M3 Ultra Studio, 96 GB UMA (see BASELINES).  
- Same frozen prompt/tool suites; cold/warm recorded.  
- Axes at minimum:
  1. **Math / formal** (Stage-1 promise)
  2. **Coding / SWE / terminal** (base + Kimi promise)
  3. **Agentic / tool-use / HCLI** (Kimi + HCLI surface)
  4. **General reasoning / knowledge**
  5. **Long-context management**
  6. **Reliability** (refusals, collapse probes, verifier pass rate for D)
  7. **Runtime** (TTFT, tok/s, active-bytes/token, peak UMA, swap)
  8. **Storage** (on-disk size, restart cost)
- **Promote C over B** only if strategic axes rise and math/base do not
  fall outside agreed tolerance.  
- **Promote D over C** only if formal/verifier/sandbox axes rise and
  coding/agentic/general do not fall outside tolerance.  
- Numbers for "tolerance" are **owner-set before the run**, not after.

### Additive-not-subtractive law

```
capability(D)  should dominate  capability(C)  on formal/reliability
capability(C)  should dominate  capability(B)  on strategic/agentic
capability(B)  should dominate  capability(A)  on math
AND
no stage may "pay for" its gain by permanently deleting a prior stage's win
```

Practical consequences:

- Stage-2 must include **math regression probes** from Stage-1.  
- Stage-3 must include **coding + agentic regression probes** from Stages
  1–2 and from BASE.  
- Adapters and Odyssey modules stay **reversible / content-addressed** so a
  bad stage can be rolled back without re-streaming donors.  
- Gravity recomposition may re-encode representation; it may **not** silently
  drop tool heads, chat templates, or HCLI action surfaces.

---

## 4. Flagship Sol-role positioning

**Final Frankenstein / Ramanujan = Sol** in the three-tier product:

```
        ┌─────────────────────────────────────┐
        │  Sol  — Final / Ramanujan           │
        │  strategy, hard reasoning,          │
        │  long-horizon orchestration         │
        └──────────────┬──────────────────────┘
                       │ delegates
           ┌───────────┴───────────┐
           ▼                       ▼
   ┌───────────────┐       ┌───────────────┐
   │ Terra         │       │ Luna          │
   │ DSV4F Gravity │       │ Qwen Gravity  │
   │ general body  │       │ code executor │
   └───────────────┘       └───────────────┘
```

Sol is not "the only model." It is the **flagship decision-maker** that can
call Terra/Luna (and tools) when cheaper tiers are enough. Resident-memory
policy for who is hot by default is in
`RESIDENT_MEMORY_AND_BPW_PLAN.md` (Qwen default-resident; Frankenstein
load-on-demand).

---

## 5. Fusion mechanism (shared, all stages)

Sealed as `REAL_AND_MINIMAL` in `FRANKENSTEIN_FUSION_OPERATION.json`:

**Allowed:** block-wise streaming distillation via latent bridge + residual
adapters + projections H_d → 4096.  

**Impossible / prohibited:**

| Name | Why |
|------|-----|
| Stream portion and average weights | Shapes mismatch (H, L, vocab, MoE layout) |
| Direct weight transplant | Bridge contracts set `direct_weight_transplant=false` |
| Hold two donors resident | Disk contract: max 1 donor window |

Inheritance payloads after a block seals: verified behavioural traces,
method capsules, verifier dispositions, repair pairs, route-aware rollout
targets, sealed reversible adapters — **not** raw donor weights/logits/KV.

---

## 6. Capability programme (what each generation must prove)

### A — BASE (DeepSeek-V4-Flash Gravity)

- Full load + forward + first token + HCLI surface + numeric parity where
  defined.  
- True served TPS when resident (streamed-layer TPS is **not** served TPS —
  runtime accounting is explicit on this).  
- Baseline scores on coding/agentic/general suites.

### B — PROTO

- Math suite lift vs A.  
- No semantic-collapse on trivial probes (see BPW plan: GLM sub-1 BPW
  collapse is the cautionary tale — Proto must not ship a "math preserve"
  that fails `2 + 2`).  
- Adapter archive sealed; GLM windows evictable.

### C — FINAL

- Strategic/agentic/coding-breadth lift vs B.  
- Math suite holds vs B within tolerance.  
- Context-management probes improve or hold.  
- Still a **single** DeepSeek-family executable child + adapters.

### D — RAMANUJAN

- Q-gauntlet progress under owner authority (Q0 historical container closed;
  Q1+ as contracts allow).  
- Verifier-dispositioned training only; no self-promoted generators.  
- Sandbox qualification: restricted execution still clears capability gates.  
- **Regression battery:** coding + agentic + tool-use + general from C must
  not be erased. Formal strength is an **add**, not a replace.

---

## 7. Programme phase map (executable overview)

| Phase | Stage | Artefact | Manual? |
|------|-------|----------|---------|
| P0 | — | Plan seals (this doc set) | no |
| P1 | 1 | Proto build + seal | eng |
| P2 | 1→2 | Clear storage | eng |
| P3 | 2 | Kimi stream + Final seal | eng |
| P4 | 2 | Four-way A/B/C ablation (D later) | eng |
| P5 | post-C | **Cloud seal Final** | **HUMAN** |
| P6 | post-C | Re-download clean DSV4F + Qwen3-Coder-30B | eng after human |
| P7 | post-C | Gravity Terra + Luna; HCLI wire | eng |
| P8 | 3 | Odyssey / Q-gauntlet / sandbox → Ramanujan | eng + owner gates |
| P9 | 3 | Four-way A/B/C/**D** full comparison | eng |

Detail for P3 → Stage-2 plan.  
Detail for P5–P7 → Post-Final plan.  
Detail for memory at P6–P7 → Resident-memory plan.

---

## 8. Model card pins (authoritative numbers)

| Model | Total | Active | Context | Card |
|-------|------:|-------:|--------:|------|
| DeepSeek-V4-Flash | 284B | 13B | 1M | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash |
| Kimi-K3 | 2.8T | 104B | 1M | https://huggingface.co/moonshotai/Kimi-K3 |
| Qwen3-Coder-30B-A3B | 30.5B | 3.3B | (card) | https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct |

Local revision pins:

- DSV4F: `60d8d70770c6776ff598c94bb586a859a38244f1`  
- Kimi-K3: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`  
- GLM-5.2 (Stage-1 donor): `b4734de4facf877f85769a911abafc5283eab3d9`

---

## 9. Non-goals of this programme doc

- No implementation in this lane.  
- No streaming of Kimi or GLM from the planning worktree.  
- No redesign of the three-stage stack.  
- No inventing resident-memory numbers — measure on the 96 GB box.

---

## 10. Cross-references

- Fusion op seal: `workspace/campaign/evidence/models/frankenstein/FRANKENSTEIN_FUSION_OPERATION.json`  
- Pipeline (K3 admitted): `workspace/campaign/evidence/models/deepseek-v4/DEEPSEEK_V4_FRANKENSTEIN_PIPELINE_PLAN_K3_ADMITTED.json`  
- Odyssey substrate caution: `workspace/campaign/governance/odyssey/program/launch/SUBSTRATE_CAPABILITY.json`  
- Q0–Q6: `ramanujan/governance/contracts/RAMANUJAN_Q0_Q6_CONTRACTS.json`  
- Roles economics: `ramanujan/governance/contracts/RAMANUJAN_ROLES_ECONOMICS.json`
