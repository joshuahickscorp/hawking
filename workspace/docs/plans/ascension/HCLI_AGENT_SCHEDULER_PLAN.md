# HCLI Agent Scheduler Plan

**Status:** PLAN + SCAFFOLD — build-ready, gated, model-free.  
**Bible:** Ascension §12 (Agent OS overview), §13 (Agent Scheduler), §25 (Residency modes).  
**Gate:** Programme step “activate HCLI Agent OS” runs **after** Proto-Frankenstein is sealed, offloaded, hash-verified, and removed from active local storage (bible §0). **No Qwen download, Gravity work, or model execution in this lane.**  
**Scaffold module:** `crates/hide-backend/src/agent_scheduler.rs`  
**Schema:** `hcli.agent_scheduler.v1`

---

## 1. Problem

HCLI must run **many logical agents** against **one loaded model weight copy** (bible §13):

```text
sessions · queues · priorities · residency · continuous batching
tool-wait suspension · checkpoint/resume · fairness · starvation prevention
```

Do **not** create one weight copy per agent.

Measures (later, when integrated):

| Metric | Owner when live |
|--------|-----------------|
| verified tasks/hour | host + verification plane |
| per-agent p99 | agent scheduler metrics |
| aggregate accepted TPS | hawking-serve batch |
| queue time | `SchedulerMetrics.total_queue_wait_ms` |
| tool wait | `SchedulerMetrics.total_tool_wait_ms` |
| GPU utilization | serve / Metal counters |
| memory pressure | fleet `FleetGovernor` + OS probe |

---

## 2. What already exists (reuse — do not reinvent)

| Primitive | Crate / path | Role |
|-----------|--------------|------|
| `PriorityClass`, `ConcurrencyClass`, `AgentJob`, `JobGraph` | `hide-fleet::queue` | Machine-wide **job** queue (event-log projection) |
| `FleetGovernor`, `FleetScheduler`, `TickPlan`, `PoolOccupancy` | `hide-fleet::scheduler` | RAM / thermal / spawn-rate **admission** for jobs |
| `FleetManager` | `hide-fleet::manager` | Launch kernel runs under fleet policy |
| `SessionId`, `RunId`, `SessionRegistry` | `hide-core` / `hide-backend::services_session` | Durable session identity |
| `AgentState`, `Phase` FSM, `AgentKernel::step` | `hide-kernel::machine` | **Single** agent run driver |
| `AgentCheckpoint` | `hide-kernel::checkpoint` | Snapshot / restore / fork at log seq |
| Continuous batch slots, `BatchPolicy`, `PrefixIndex` | `hawking-serve::batch` | **Token-level** multi-request decode |
| Role routing / energy admission | `hawking-orch::{router,scheduler}` | Which model role to call under power budget |
| HCLI parallel lanes | `hide-backend::hcli_swarm` | Outer multi-kernel analysis (not shared-weight) |

### Layering (settled)

```text
┌─────────────────────────────────────────────────────────┐
│ hide-backend::agent_scheduler  (THIS SCAFFOLD)          │
│  many LogicalAgent → one ModelResidency                 │
│  queue / priority / tool-wait / fairness / preempt      │
└───────────────────────────┬─────────────────────────────┘
                            │ admits generation rights
┌───────────────────────────▼─────────────────────────────┐
│ hide-fleet                 machine jobs + resource gov  │
│ hawking-serve::batch       continuous token batching    │
│ hide-kernel                one AgentState step loop     │
└─────────────────────────────────────────────────────────┘
```

`agent_scheduler` answers: *which logical agents may hold a generation slot on a shared loaded model right now?*  
It does **not** replace fleet job admission or serve decode packing.

---

## 3. Scaffold design (implemented now)

### 3.1 Identity

- `LogicalAgentId` — one logical agent (many per residency).
- `ModelResidencyId` + `ModelResidency` — **one** loaded model process / weight residency; `attached: Vec<LogicalAgentId>`.
- `CheckpointRef` — `{ session_id, run_id, seq }` pointing at kernel checkpoint store (no byte copy in scheduler).

### 3.2 Residency modes (bible §25, agent-facing)

| Mode | Meaning in scaffold |
|------|---------------------|
| `DualResident` | Executor + reviewer co-resident; pipeline N+1 / review N |
| `ExecutorResident` | Only executor stays loaded; reviews queue |
| `PhaseSeparated` | **Default** until dual footprint is measured |

Distinct from MoE GPU weight-cache residency in `hawking-core`.

### 3.3 State machine

```text
Registered → Queued → Admitted → Generating ⇄ ToolWaiting
                             ↘ Checkpointed → Queued (resume)
Generating / ToolWaiting → Completed | Failed | Cancelled
Queued / Admitted / Generating → Preempted → Checkpointed
```

**Load-bearing rule:** `ToolWaiting` does **not** hold a batch slot (`holds_batch_slot` is only `Admitted | Generating`). A tool-suspended agent free the model for peers.

### 3.4 Tick policy (`schedule_tick`)

Order (pure, host supplies `at_ms`):

1. **Starvation boost** — if `queue_wait ≥ starvation_threshold_ms`, raise `effective_priority` one class toward Interactive.
2. **Fairness yield** — generating agent with `fairness_quantum_remaining == 0` while others wait → re-queue.
3. **Preempt** — Interactive waiting + batch full → checkpoint lowest preemptible (priority weaker than High).
4. **Admit** — ready set sorted by `(effective_priority, queued_at_ms, id)` up to free slots / `max_admit_per_tick`.

Reuses `hide_fleet::PriorityClass` and `ConcurrencyClass` (Model vs CpuOnly). CpuOnly agents do not consume model batch slots in this scaffold.

### 3.5 Metrics scaffold

`SchedulerMetrics`: admits, preemptions, starvation_boosts, fairness_yields, checkpoints, queue/tool wait ms, verified_tasks_completed.

---

## 4. What is scaffolded vs not implemented

| Capability | This pass | Later integration |
|------------|-----------|-------------------|
| Types + state transitions | **Real code + tests** | — |
| Fairness / starvation / tool-wait slot release | **Real code + tests** | Wire quanta to actual tokens |
| Checkpoint ref tracking | **Scaffold** | Call `AgentCheckpoint::snapshot` + host store |
| Continuous batching | **Cohort field only** | Map admits → `hawking-serve` slot admit |
| Weight load / unload | **Not started** | Supervisor + residency modes A/B/C |
| Kernel `step` drive | **Not started** | `FleetManager` / live thread per agent |
| Live metrics export | **Counters only** | Wire-B projection / Prometheus |
| Multi-residency (30B+80B dual) | **Data model only** | Phase-separated unload orchestrator |

---

## 5. Explicit non-goals (this pass)

- No Qwen / Gravity / model download or inference.
- No modification of `lab/operators/frankenstein_*`, Frankenstein evidence trees, or DSV4F/GLM capture paths.
- No push / PR / remote.
- No claim that multi-agent generation is “implemented” end-to-end — only the **scheduler policy layer** is real and tested.
- No new weight-copy-per-agent design (forbidden by bible).

---

## 6. Integration checklist (when Agent OS activates)

1. Host opens `ModelResidency` when `hawking serve` loads a role; tear down on unload.
2. Each HCLI session/turn registers a `LogicalAgent` bound to that residency.
3. On kernel `Act` needing generation: ensure agent is `Generating`; on tool dispatch: `suspend_for_tool` **before** releasing serve slot.
4. On fleet preempt / thermal: `checkpoint` → kernel snapshot → `resume_from_checkpoint` after re-admit.
5. Feed `consume_quantum` from accepted tokens (or step count) so fairness is real.
6. Publish `SchedulerMetrics` beside fleet occupancy in status RPC.
7. Dual-resident pipeline only after measured footprint says 30B+80B+target fit.

---

## 7. Tests (scaffold)

Module tests in `agent_scheduler.rs` cover:

- many agents / one residency
- tool-wait releases batch slot
- tool-wait accounting
- checkpoint / resume
- interactive preempts batch when full
- starvation boost
- fairness yield after quantum
- illegal transitions / terminal refuse
- priority order Interactive before Batch

Run: `cargo test -p hide-backend --lib agent_scheduler`

---

## 8. Files

| Path | Role |
|------|------|
| `crates/hide-backend/src/agent_scheduler.rs` | Scaffold implementation + tests |
| `crates/hide-backend/src/lib.rs` | Module export |
| `workspace/docs/plans/ascension/HCLI_AGENT_SCHEDULER_PLAN.md` | This plan |
