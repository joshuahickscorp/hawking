# HCLI Skill Foundry Plan (Ascension Bible §18)

**Status:** PLAN + SCAFFOLD — gated on Proto-Frankenstein offload; no live model work.  
**Scaffold code:** `crates/hawking-context/src/skill_foundry.rs`  
**Companion:** `HCLI_MEMORY_OS_PLAN.md` (admitted skills → Memory OS **L3 SKILLS**)  
**Programme gate:** future Agent OS slice after Memory OS bridge; sandbox never holds final authority.

---

## 1. Purpose

A repeated successful workflow may be **proposed** as a skill. Skills must be:

```text
versioned
tested
source-bound
retrievable
composable
retirable
```

**Authority boundary (bible):**

```text
sandbox may propose
only the protected controller may admit
```

Every evolution requires **replay**, **hidden validation**, and **compatibility testing** before protected admission.

This plan scaffolds real types + a deterministic admission pipeline. It does not run live models, Gravity, or frankenstein paths.

---

## 2. Audit — what already exists vs Foundry

| Existing | Role | Relation to Foundry |
|----------|------|---------------------|
| `MemoryClass::Procedural` | Durable successful recipes / tool receipts | **Pre-skill** store; Foundry elevates verified procedures into versioned skills |
| `ProceduralWriteCap` | Write authority for recipes | Maps to post-admit L3 write path |
| `personal_tools` | Tool ABI + propose → permission gate → single-use receipt → execute | Pattern for **gated execution**; not skill admission |
| `hide-fleet` / TQ `Admission` | Resource / GPU admission | Unrelated name collision — do not reuse |
| Claude / Grok `SKILL.md` frontmatter | Human-authored skill packs on disk | Prior art for name/description/procedure docs; Foundry is **admit-gated runtime registry** |

**Gap:** no skill schema, no propose→admit pipeline, no protected-controller type boundary for skills, no composability/compatibility registry.

---

## 3. Skill schema (bible §18)

```text
name
purpose
scope
inputs
outputs
preconditions
procedure          # ordered steps; source-bound via source_ref
environment        # platforms, tools, env var names, notes
provenance         # proposed_by, source_workflow_ids, evidence_refs, success_count
tests
failure_modes
compatibility      # composes_with, conflicts_with, min_foundry_schema, surfaces
version            # major.minor.patch
```

Example name from bible: `optimize_qwen_moe_projection_wave` (schema-only; no live run in this slice).

Scaffold types: `SkillSpec`, `SkillIoField`, `SkillStep`, `SkillEnvironment`, `SkillProvenance`, `SkillTest`, `SkillFailureMode`, `SkillCompatibility`, `SkillVersion`.

### Propose-time validation (hard requirements)

- Non-empty `name`, `purpose`  
- ≥1 procedure step  
- ≥1 test (bible: **tested**)  
- Source-bound: ≥1 `source_workflow_ids` **or** `evidence_refs`  

---

## 4. Admission path

```text
propose
  → replay
  → hidden validation
  → compatibility test
  → protected admission
```

| Stage | Who | Scaffold behaviour |
|-------|-----|--------------------|
| **Propose** | Sandbox (`SandboxProposeCap`) | Validate schema; status=`proposed`; not retrievable |
| **Replay** | Pipeline | Procedure steps have non-empty actions (stub; later: re-run receipts) |
| **Hidden validation** | Pipeline | Declared tests well-formed (stub; later: holdout fixtures, no sandbox peek) |
| **Compatibility test** | Pipeline | Fail if `conflicts_with` intersects admitted skill names; schema prefix check |
| **Protected admission** | Controller only (`ProtectedControllerCap`) | Requires all prior stage pass receipts + non-empty controller id |

### Capability type boundary

Same pattern as `VerifierWriteCap` / `UserWriteCap`:

- `SandboxProposeCap` — mint on sandbox path; **only** `propose`.  
- `ProtectedControllerCap` — mint only on protected controller entry; **required** by `admit`.  
- Sandbox code that tries to call `admit` without the cap type is a compile-time / review-visible violation.

### Lifecycle statuses

```text
proposed → replaying → validating → compatibility_testing → admitted
                                                              ↘ retired
any failing stage → rejected
```

**Retrievable as skill** ⇔ `status == admitted` (L3 Memory OS).

---

## 5. Design decisions

1. **Foundry is not personal_tools.** Tools execute under permission receipts; skills are **reusable procedures** admitted into L3. A skill may *invoke* tools; admitting a skill is not executing one.  
2. **Procedural memory remains the raw substrate.** Distillation path: N successful procedural receipts → sandbox `propose` → pipeline → controller `admit` → Memory OS L3 item + optional procedural supersession.  
3. **Retire ≠ forget.** Retired skills stay inspectable for audit; drop from default retrieve. New version = new proposal (semver bump), not silent mutate-in-place of an admitted skill.  
4. **Composability** is declared (`composes_with`) and conflict-checked; full graph planning is later work.  
5. **No dual authority.** Sandbox output alone never sets `Admitted`.

---

## 6. Integration with Memory OS

| Event | Memory OS effect |
|-------|------------------|
| Skill admitted | Store L3 item: text=purpose+procedure summary, source=`skill_foundry:{id}`, verification=`Proven` (controller-attested), tags=`skill:{name}`, `version:…` |
| Skill retired | Invalidate or graveyard L3 item with reason; Foundry status=`retired` |
| Skill rejected | No L3 write; optional L1/L4 note for learning |

See `HCLI_MEMORY_OS_PLAN.md` phase M6.

---

## 7. Scaffold inventory

| Path | Role |
|------|------|
| `crates/hawking-context/src/skill_foundry.rs` | Schema, caps, pipeline stub, tests |
| `crates/hawking-context/src/lib.rs` | Module + re-exports |
| This plan | Design + audit |

Public helpers: `example_skill_spec`, `SkillFoundry::run_admission_pipeline`.

Schema id: `hide.skill_foundry.v0`.

---

## 8. Tests (scaffold)

- Sandbox proposes; admit without prior stages denied  
- Full pipeline admits under protected controller; stage receipts complete  
- Empty procedure / unbound source rejected at propose  
- Compatibility conflict with admitted skill → rejected  
- Retire removes from retrievable list  
- Version + composability serde roundtrip  
- Stages must run in order  

---

## 9. Remaining work (post-gate)

| Phase | Work |
|-------|------|
| S1 (done scaffold) | Types + stub pipeline + caps + tests |
| S2 | Persist Foundry registry (SQLite alongside classed memory) |
| S3 | Real replay against recorded workflow receipts / tool traces |
| S4 | Hidden validation harness (fixtures not visible to proposing sandbox) |
| S5 | Host commands: `skill.propose`, `skill.admit`, `skill.list`, `skill.retire` |
| S6 | Bridge admitted skills → Memory OS L3 + procedural supersession |
| S7 | Composition planner (ordered multi-skill runs) |
| S8 | Evolution path: change skill → force full pipeline again |

---

## 10. Non-goals (this slice)

- Live Qwen / Gravity skill induction  
- Touching `lab/operators/frankenstein_*` or frankenstein evidence  
- Automatic admission without protected controller  
- Detached daemons, push/PR, venv commits  

---

## 11. Confidence

| Claim | Confidence |
|-------|------------|
| No prior Skill Foundry in-repo; scaffold is net-new | **High** |
| Procedural memory is the right pre-skill substrate | **High** |
| Cap-typed admit boundary matches existing security style | **High** |
| Stub replay/validation is enough until Agent OS is live | **High** for plan; **Low** as production gate |
| Full receipt-replay fidelity needs S3–S4 | **High** that work remains |
