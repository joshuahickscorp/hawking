# HCLI Planning Diagnostic Plan

**Status:** PLAN + SCAFFOLD — build-ready, gated, model-free.  
**Bible:** Ascension §12 (Agent OS overview), §14 (Planning Diagnostic).  
**Gate:** Same Agent OS activation gate as §0 / §35 step 9 — **after** Proto-Frankenstein is sealed off local active storage. **No model plan synthesis in this lane.**  
**Scaffold module:** `crates/hide-backend/src/planning_diagnostic.rs`  
**Schema:** `hcli.planning_diagnostic.v1`

---

## 1. Problem

Separate planning into an explicit, receipted pipeline (bible §14):

```text
GOAL INTERPRETATION
→ TOOL RETRIEVAL
→ PLAN
→ PLAN CHALLENGE
→ EXECUTION
→ OBSERVATION
→ REPLAN
→ VERIFICATION
→ REPORT
```

Each stage has a **receipt**.

For **kernel / research** planning, the planner **must** answer:

| Question | Contract field |
|----------|----------------|
| What is the measured bottleneck? | `measured_bottleneck` |
| What evidence distinguishes candidate explanations? | `distinguishing_evidence` |
| What is the cheapest experiment that can disprove the hypothesis? | `cheapest_disprove_experiment` |
| Which tools are required? | `required_tools` |
| What result causes promotion? | `promotion_result` |
| What result causes retirement? | `retirement_result` |

**Never allow “try optimizations” as an unbounded plan** — encoded as schema validation (`UNBOUNDED_PLAN_MARKERS` + `ExperimentBound`).

---

## 2. What already exists (reuse — do not reinvent)

| Primitive | Crate / path | Role |
|-----------|--------------|------|
| `Plan`, `PlanStep`, `Acceptance`, `StepKind` | `hide-kernel::plan::schema` | Execution DAG; every step declares acceptance oracles (K1) |
| `PlanDag`, `Planner` trait, `StubPlanner`, `RuntimePlanner` | `hide-kernel::plan` | Synthesize / replan execution plans |
| Kernel `Phase` (`Intake`…`Plan`…`Replan`…`Done`) | `hide-kernel::machine::state` | Run driver FSM (not the diagnostic protocol) |
| `PlanRecord`, `PlanRecordStore` | `hide-backend::plan_domain` | Durable IDE PlanCard projection |
| Research pipeline / CAS receipts | `hawking-research` | Evidence pin + research run FSM (adjacent) |
| Headless / swarm receipts | `hide-backend::{headless,hcli_swarm}` | Outer audit receipt shapes |

### Relationship (settled)

```text
PlanningDiagnosticRun          ← quality protocol + stage receipts (THIS)
        │ produces / challenges
        ▼
hide_kernel::plan::Plan        ← executable DAG + acceptance oracles
        │ projected for UI as
        ▼
plan_domain::PlanRecord        ← PlanCard
        │ driven by
        ▼
AgentKernel Phase FSM          ← Act / Observe / Verify / Repair
```

The diagnostic does **not** replace the kernel plan schema. It wraps **plan quality** (especially research) before execution is allowed.

---

## 3. Scaffold design (implemented now)

### 3.1 Stages

`DiagnosticStage` enum with `wire_name()`, `successors()`, and legal transition checks. Replan loops are first-class (`Observation|Verification → Replan → PlanChallenge`).

### 3.2 Receipts

`StageReceipt`:

- `stage`, `status` (`Pending|Running|Passed|Failed|Skipped|Blocked`)
- `summary`, structured `payload`
- `started_ms` / `finished_ms`, optional `error`

Every stage that runs leaves a receipt on `PlanningDiagnosticRun.receipts`.

### 3.3 Kernel research contract

`KernelResearchContract` + `ExperimentBound { max_steps, max_wall_ms, max_compute_units }`.

`validate()` enforces:

- non-empty answers for all six bible questions
- non-empty `required_tools`
- non-zero experiment bound
- promotion ≠ retirement
- disprove text does not contain unbounded markers

### 3.4 Unbounded language gate

`reject_unbounded_language` scans plan title/objective/steps for markers including:

- `try optimizations` / `try some optimizations` / `try various optimizations`
- `optimize until better` / `keep optimizing` / `tune until`
- `improve performance somehow` / `randomly try` / `just try things` / `see what works`

Works in **both** `PlanningMode::General` and `KernelResearch`. Research mode **additionally** requires a valid contract before PlanChallenge can pass.

### 3.5 Challenge / execution gates

- `resolve_challenge(accepted)`: on reject → stage `Blocked`, return to `Plan` (run stays Active).
- `enter_execution` in `KernelResearch` requires a **passed** PlanChallenge receipt.
- `replan` requires a prior Observation receipt and respects `max_replans`.
- `complete_verification` / `emit_report` close the pipeline.

### 3.6 Fixture

`fixture_kernel_research_happy_path` walks the full pipeline with a valid contract (model-free demo / test oracle).

---

## 4. What is scaffolded vs not implemented

| Capability | This pass | Later integration |
|------------|-----------|-------------------|
| Stage enum + legal transitions | **Real + tests** | — |
| Per-stage receipts | **Real + tests** | Persist to event log / KV |
| Unbounded-plan rejection | **Real + tests** | Surface in PlanCard UI |
| Kernel research contract schema | **Real + tests** | Fill from profiler / ledger |
| Plan challenge gate | **Real + tests** | Model critic / second role |
| Goal interpretation / tool retrieval | **Receipt slots only** | Retrieval gateway (§15) + tool gateway (§16) |
| Model plan synthesis | **Not started** | `RuntimePlanner` + dual-Qwen |
| Map proposal → kernel `Plan` DAG | **Not started** | Planner adapter |
| Map → `PlanRecord` for IDE | **Not started** | `plan_domain::from_kernel` after synthesize |
| Execution / observation drive | **Not started** | Agent scheduler + kernel step |
| Promotion / retirement ledger write | **Contract text only** | Science registry / campaign receipts |

---

## 5. Explicit non-goals (this pass)

- No model-backed plan generation or plan critique.
- No automatic experiment execution against Metal kernels.
- No Qwen / Gravity / Frankenstein / GLM capture work.
- No claim that “planning diagnostic is live in the agent loop” — only the **schema, gates, and receipt state machine** are real.
- Do not weaken kernel K1 (acceptance oracles on every execution step); diagnostic is additive.

---

## 6. Integration checklist (when Agent OS activates)

1. Host starts `PlanningDiagnosticRun` for research / kernel goals (`PlanningMode::KernelResearch`).
2. Goal interpretation stage writes measured objective + scope into receipt payload.
3. Tool retrieval stage consults Tool Index (§16) and records tool ids in receipt.
4. Plan stage: model or heuristic produces `DiagnosticPlanProposal`; `attach_proposal` validates.
5. Plan challenge: reviewer role or deterministic critic; `resolve_challenge`.
6. On accept: compile proposal → `hide_kernel::plan::schema::Plan` with acceptance oracles; publish `PlanRecord`.
7. Execution: agent scheduler admits logical agent; kernel drives steps; observation receipt mirrors verify outcomes.
8. Promotion / retirement: write campaign science receipts using contract thresholds (not free-form “looks better”).
9. Report stage seals operator-facing summary; never grant sandbox models final authority (bible §0 / verification authority).

---

## 7. Tests (scaffold)

Module tests in `planning_diagnostic.rs` cover:

- reject `try optimizations` language
- empty contract fields / unbounded experiment / promotion==retirement
- kernel research requires contract; general mode does not
- full happy-path fixture emits all stage receipts
- challenge rejection returns to Plan with Blocked receipt
- execution blocked without passed challenge in research mode
- replan needs observation + respects budget
- illegal stage skip rejected

Run: `cargo test -p hide-backend --lib planning_diagnostic`

---

## 8. Files

| Path | Role |
|------|------|
| `crates/hide-backend/src/planning_diagnostic.rs` | Scaffold implementation + tests |
| `crates/hide-backend/src/lib.rs` | Module export |
| `workspace/docs/plans/ascension/HCLI_PLANNING_DIAGNOSTIC_PLAN.md` | This plan |
