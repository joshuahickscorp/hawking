# HCLI Option-C Sandbox Plan

**Status:** PLAN + SCAFFOLD ONLY — future programme, gated on Proto-Frankenstein offload.  
**Bible:** HAWKING_ASCENSION_BIBLE §24 (Option-C sandbox), §21–22 (sandbox + verification authority).  
**Scaffold:** `lab/hcli/option_c.py`  
**Tests:** `lab/tests/test_option_c.py`  
**Reference implementation (already running as habit):** Claude + Grok delegate/audit loop via `grok-orchestration`.

---

## 1. Purpose

Option-C is the standing dual-model research sandbox for HCLI after bootstrap:

```text
30B  → executor
80B  → reviewer
protected controller → authority (parity, held-out, CLEAN, sign promote/rollback)
```

It is **logical, not necessarily simultaneous**. Phases may be serialised under
residency Modes B/C (`HCLI_RESIDENCY_MODES_PLAN.md`).

**Critical recognition:** tonight's Claude + Grok orchestration **is already a
working Option-C instance**. This plan formalises that pattern as a subsystem
spec, not a one-off habit.

---

## 2. Structural identity with tonight's pattern

| Option-C role | Tonight's reference | Isolation | May edit? | Output |
|---------------|---------------------|-----------|-----------|--------|
| **30B executor** | `grok-run delegate` | git worktree `grok/<task-id>` | yes (owned worktree only) | candidate report + diff (claim) |
| **80B reviewer** | `grok-run audit` | kernel-enforced read-only sandbox | **no** | review findings (independent challenge) |
| **Protected controller** | Claude (architect + final authority) | full tree + protected oracles | repair/merge only after decision | signed promote / rollback / hold |

### Authority that never transfers

From `grok-orchestration/SKILL.md` and `delegate.md`:

- Grok's report is a **claim**, not a receipt. Controller verifies the artifact.
- Executor never pushes, merges, deploys, or signs promotion.
- Reviewer never receives controller conclusions before reviewing (independence fence).
- Controller synthesises: agreements first, disagreements investigated, findings
  marked confirmed / unconfirmed / refuted with evidence.

### Skill files that encode the habit

| File | Option-C analogue |
|------|-------------------|
| `~/.claude/skills/grok-orchestration/SKILL.md` | role split, non-negotiables, profiles |
| `delegate.md` | executor contract, worktree isolation, review checklist, disposition |
| `audit.md` | independent / adversarial audit; no code changes; synthesis rules |
| `grok-run` wrapper | worktree lifecycle, artifacts (`grok-report.md`, `diff.patch`), cleanup |

The scaffold's `OptionCSandbox.role_map_to_tonight()` freezes this mapping as a
machine-readable receipt.

---

## 3. Workflow (logical)

```text
30B executor
    inspect
    hypothesize
    edit isolated worktree
    compile
    run allowed tests
    emit candidate report

80B reviewer
    independently inspect          ← no controller conclusions
    challenge parity
    challenge benchmark
    challenge architecture
    request distinguishing tests
    emit review report

protected controller
    run protected parity           ← Numeric Parity V2.1 family
    run held-out capability
    run CLEAN benchmark
    sign promotion or rollback     ← only signer
```

### Candidate phase machine (scaffold)

```text
IDLE
 → EXECUTING
 → CANDIDATE_EMITTED
 → REVIEWING
 → REVIEW_EMITTED
 → CONTROLLER_EVAL
 → PROMOTED | ROLLED_BACK | REJECTED | INSUFFICIENT_EVIDENCE
```

Sequential by construction; co-residency is a residency-mode concern, not an
Option-C requirement.

---

## 4. Mandatory review

Independent 80B review is **required** before controller decision for:

| Category | Rationale |
|----------|-----------|
| `kernel_promotion` | production kernel surface |
| `quantization_change` | numeric / capability risk |
| `routing_change` | MoE / scheduler behaviour |
| `benchmark_change` | evaluation integrity |
| `runtime_scheduling` | fairness / starvation / safety |
| `storage_deletion` | irreversible local loss |
| `artifact_promotion` | durable serve path |
| `effect_authority` | who may cause external effects |

Controller `begin_eval` refuses when a mandatory category lacks a review report.
Non-mandatory categories may skip review (controller still runs protected gates).

---

## 5. May / may not (executor)

**May:** read source, inspect public profiles, edit owned worktree, compile, run
allowed tests, request protected benchmark, request approved downloads, emit
candidate report.

**Must not:** modify protected oracle, modify held-out prompts, modify promotion
thresholds, merge self, sign own results, delete stable artifacts, read/print
credentials.

Reviewer **must not** edit worktrees, merge, sign promotion, delete artifacts, or
see controller conclusions pre-review.

Encoded as `EXECUTOR_ALLOWED` / `EXECUTOR_FORBIDDEN` / `REVIEWER_*` in the scaffold.

---

## 6. Controller decision honesty

Mirrors `frankenstein_promotion_gate.evaluate_promotion`:

| Gate evidence | Decision |
|---------------|----------|
| any FAIL | REJECT (or ROLLBACK if already staged) |
| all PASS + reviewer not REJECT | PROMOTE |
| any PENDING / missing | **HOLD** → `INSUFFICIENT_EVIDENCE` |
| force PROMOTE with FAIL/PENDING | refused |

Result classes (Bible §22): `PROMOTED_MECHANISM`, `REJECTED_MECHANISM`,
`TOOL_DEFECT`, `PLANNING_DEFECT`, `VERIFIER_DEFECT`, `ENVIRONMENT_DEFECT`,
`INSUFFICIENT_EVIDENCE`.

Sandbox roles can never be `signed_by`.

---

## 7. Interface (scaffold)

```text
lab/hcli/option_c.py
  Role, CandidatePhase, MANDATORY_REVIEW_CATEGORIES
  CandidateReport, ReviewReport, ControllerDecision
  OptionCSandbox, OptionCController
```

Schemas: `hawking.hcli.option_c.v1`, `.candidate.v1`, `.review.v1`, `.decision.v1`.

Bootstrap model IDs (logical, no load):

- Executor: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- Reviewer: `Qwen/Qwen3-Coder-Next` (80B-class)

---

## 8. Phase plan (future)

### OC.0 — Scaffold (this revision)

- [x] Role map to grok-orchestration  
- [x] Phase machine + mandatory review fence  
- [x] Controller HOLD/PROMOTE/REJECT honesty  
- [x] Tests  

### OC.1 — Bind executor to worktree runtime

- Reuse `grok-run delegate` semantics inside HCLI: isolated branch, artifact dir,
  cleanup discipline (`cleanup` in the accept/reject turn).  
- Still no remote push/merge from sandbox roles.

### OC.2 — Bind reviewer to read-only audit runtime

- Kernel-enforced read-only (Seatbelt or equivalent).  
- Criteria-first contracts; no executor conclusions in the prompt.

### OC.3 — Protected gate adapters

- Wire real Numeric Parity V2.1, held-out capability, CLEAN benchmark runners.  
- Until then, controller continues to return HOLD on missing bundles.

### OC.4 — Residency coupling

- Mode A: pipeline execute N+1 while review N.  
- Mode B: queue reviews until target unloads.  
- Mode C: full phase separation (see residency plan).

### OC.5 — Self-evolution handoff

- Promoted mechanisms may open self-evolution proposals; admission remains a
  separate protected path (`HCLI_SELF_EVOLUTION_PLAN.md`).

---

## 9. Non-negotiables

- Structural identity with Claude-controller + Grok-delegate + Grok-audit.  
- Logical Option-C (no requirement for simultaneous 30B+80B residency).  
- Mandatory review list is closed and controller-enforced.  
- No self-merge, no self-sign, no oracle/threshold edits by sandbox models.  
- No fabricated PROMOTE.  
- No live Qwen load in this scaffold revision.  
- Do not disturb frankenstein live recapture paths.

---

## 10. Exit criteria for "Option-C live"

1. OC.1–OC.3 complete with sealed decision receipts.  
2. At least one mandatory-category candidate promoted and one rejected under review.  
3. Independent audit confirms reviewer never saw controller conclusions.  
4. Role map receipt still matches operational orchestrator (or is versioned when it changes).
