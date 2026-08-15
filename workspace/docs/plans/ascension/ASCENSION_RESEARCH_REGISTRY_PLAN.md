# Ascension Research Registry Plan

**Status:** plan + scaffold only  
**Bible:** HAWKING_ASCENSION_BIBLE §4  
**Scaffold:** `lab/operators/research_registry.py`  
**Shape reference:** `workspace/campaign/evidence/models/frankenstein/FRANKENSTEIN_ARCHITECTURE_OPTIONS.md`

---

## 1. Purpose

Do not begin by downloading full models and later discover that Gravity layouts
fight the kernels. Before final Qwen artifacts are created, run bounded research
using official architecture/config metadata, small source windows, representative
tensors, synthetic geometry fixtures, small compatible family models, CPU
authorities, Metal microbenchmarks, and existing Hawking negative-science ledgers.

Every research item ends as exactly one of:

```text
ADMIT_TO_GRAVITY
ADMIT_TO_RUNTIME
ADMIT_TO_KERNEL
DEFER
REJECT
```

Every result records:

```text
mechanism
hypothesis
expected B/F/U/R/D/S/K effect
source geometry
prototype
measured result
capability risk
Gravity implication
runtime implication
reopen condition
```

The research phase is complete only when all mechanisms that could materially
affect Gravity packing have a decision.

---

## 2. Pattern generalised from FRANKENSTEIN_ARCHITECTURE_OPTIONS.md

Tonight's real research verdict document already has the right *shape*, written
in prose tables rather than types:

| FRANKENSTEIN field | §4 registry field |
|--------------------|-------------------|
| Technique / What | `mechanism` + `hypothesis` |
| Cost / Risk / Scale / Cross-arch feasibility | `expected_bfurdsk`, `capability_risk`, `constraints_checked`, `feasibility_notes` |
| Verdict: PRIMARY / BEST V1+ / DEFER / RULED OUT / NOT APPLICABLE / POOR FIT | `verdict` enum |
| Citations + sealed local contracts | `citations`, `source_geometry`, `prototype` |
| "only worth building if V0 …" / park language | `reopen_condition` |
| Gravity / runtime notes in recommendation | `gravity_implication`, `runtime_implication` |
| Measured (often "pending V0") | `measured_result` |

Scaffold mapping helpers:

```text
PRIMARY              → ADMIT_TO_RUNTIME   (runtime path; Gravity if layout-affecting)
BEST POST-V0 UPGRADE → DEFER
HIGH RISK / DEFER    → DEFER
RULED OUT            → REJECT
NOT APPLICABLE       → REJECT
POOR FIT / OUT       → REJECT
```

When a technique changes **packing layout**, prefer `ADMIT_TO_GRAVITY`.  
When it changes **kernel grammar**, prefer `ADMIT_TO_KERNEL`.  
When it is a runtime-only adapter/bridge (V0 residual), `ADMIT_TO_RUNTIME`.

Example (from FRANKENSTEIN recommendation rank 0–3):

| Mechanism | Verdict |
|-----------|---------|
| Always-on residual bridge (V0) | ADMIT_TO_RUNTIME |
| Multi-adapter hub (external router) | DEFER (reopen after V0) |
| FuseLLM-style body distill | DEFER |
| Native guest experts / mergekit / BTX foreign | REJECT |

---

## 3. Type design (scaffolded)

```text
ResearchVerdict          enum of five Bible outcomes
BFURDSKEffect            B,F,U,R,D,S,K strings (qualitative or numeric)
ResearchItem             frozen dataclass with full §4 field set + citations
ResearchRegistry         in-memory store: record / by_verdict /
                         incomplete_mechanisms / research_phase_complete
item_from_mapping        JSON round-trip
map_frankenstein_verdict prose → enum
```

Schema ids:

- `hawking.ascension.research_item.v1`
- `hawking.ascension.research_registry.v1`

### Required field discipline

- All string fields non-empty (constructor refuses blanks)
- `REJECT` still requires an explicit `reopen_condition` (`never` only when permanently closed)
- Duplicate `item_id` refused (amend via new id or future supersede path)
- `related_graveyard_ids` links §32 burials so re-proposal is auditable

### Completeness gate

```python
registry.research_phase_complete(required_mechanisms)
```

returns true only when every named mechanism has a recorded decision — the
Bible's definition of research-phase complete for Gravity packing.

---

## 4. Workflow (post Proto-Frankenstein)

1. **Inventory mechanisms** that could affect Qwen-30B / Qwen-80B Gravity packing
   (Bible §5 portfolio: data-local packed execution, projection/expert waves,
   cross-session weight amortization, cross-token invariant reuse, delta
   projection, …).
2. **Graveyard check first** (§32 / `ascension_graveyard.check_proposal`).
3. **Bounded experiment** (fixture / small window / Metal microbench) — not full model download.
4. **Record ResearchItem** with measured_result (or explicit "unmeasured → DEFER/REJECT").
5. **Admit only with human/tribunal-style review** for Gravity/kernel admissions
   (kinship with ramanujan Tribunal: producer ≠ admitter — future wiring).
6. **Seal registry snapshot** into campaign evidence when a programme freezes.

---

## 5. Relationship to other systems

| System | Relationship |
|--------|--------------|
| `lab/science_registry.py` | Operator catalogue; research *findings* live here, not as operators |
| `ramanujan` Odyssey `ResearchBranch` | Method-family branches + bury/reopen; do not write from this scaffold |
| Ascension Graveyard | Pre-check before recording; link `related_graveyard_ids` |
| Credential broker | Research may use metadata + small windows via broker; never full Qwen body until admitted programme |
| FRANKENSTEIN_ARCHITECTURE_OPTIONS.md | Seed corpus; map into typed items when programme freezes |

---

## 6. Tests

`lab/tests/test_research_registry.py`:

- five verdicts present
- FRANKENSTEIN prose map
- record / counts / duplicate refuse
- phase completeness
- mapping round-trip
- empty mechanism refuse

---

## 7. Remaining work

1. Seed registry from FRANKENSTEIN_ARCHITECTURE_OPTIONS as sealed JSON items
2. Author Qwen-30B / Qwen-80B required-mechanism checklists (§5 portfolio)
3. Durable seal path + campaign evidence location
4. Wire Graveyard pre-check into `ResearchRegistry.record` optionally
5. Tribunal/human gate for ADMIT_TO_GRAVITY / ADMIT_TO_KERNEL
6. UI/HCLI surface: list open DEFERs with reopen conditions

---

## 8. Non-goals

- No live model training or full artifact construction from research alone
- No automatic promotion of ADMIT_* into production without human gate
- No ramanujan write-path coupling in this scaffold
