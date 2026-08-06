# HCLI Residency Modes Plan

**Status:** PLAN + SCAFFOLD ONLY — future programme, gated on Proto-Frankenstein offload.  
**Bible:** HAWKING_ASCENSION_BIBLE §25 (Residency modes).  
**Scaffold:** `lab/hcli/residency.py`  
**Tests:** `lab/tests/test_residency_modes.py`  
**Companion:** Option-C (`HCLI_OPTION_C_PLAN.md`) supplies the logical roles that these modes place in memory.

---

## 1. Purpose

Campaign continuity **does not require permanent model residency**. Three modes
trade co-residency for throughput vs peak memory:

| Mode | Fit condition | Behaviour |
|------|---------------|-----------|
| **A — dual resident** | 30B + 80B + target fit safely | Pipeline: 30B executes candidate **N+1** while 80B reviews **N** |
| **B — executor resident** | 30B + target fit; 80B does not co-reside | Queue reviews until target unloads; then drain with 80B |
| **C — fully phase separated** | nothing permanently co-resident | build → checkpoint/unload → benchmark → seal → unload → review → unload → decide → resume |

This scaffold is a **pure state machine**. It does not load Qwen/Gravity weights,
allocate Metal heaps, or talk to a runtime. Future work binds `load`/`unload` to
real handles.

---

## 2. Preconditions

| Gate | Requirement |
|------|-------------|
| Proto-Frankenstein | Offloaded / out of active envelope (same programme gate) |
| Fit measurement | Live free-bytes / working-set report → `FitReport` (stub today) |
| Option-C roles | Executor / reviewer / controller defined |
| Pressure governor | Bible §27 colours may force mode degradation (future wire-up) |

---

## 3. Mode specifications

### Mode A — dual resident

**Required loaded slots:** `executor_30b`, `reviewer_80b`, `target`.

**Operations:**

- `pipeline_assign(executing=N+1, reviewing=N)` with distinct candidate ids.  
- Partial unload **refused** while still in A (must `switch_mode` first).

**When to use:** host has headroom; maximise Option-C throughput.

### Mode B — executor resident

**Required loaded slots:** `executor_30b`, `target`.

**Operations:**

- `enqueue_review(candidate_id)` while building.  
- `load(reviewer_80b)` refused while target still resident.  
- `begin_review_drain()` unloads target (+ executor), loads reviewer, returns pending ids.  
- `complete_review` marks queue items drained.

**When to use:** 80B cannot share residency with the active target.

### Mode C — fully phase separated

**Required permanent slots:** none.

**Sub-phases (strict chain):**

```text
IDLE
 → BUILD                 (30B)
 → CHECKPOINT_UNLOAD
 → BENCHMARK             (target)
 → SEAL_EVIDENCE         (target)
 → UNLOAD_TARGET
 → REVIEW                (80B)
 → EMIT_REVIEW           (80B)
 → UNLOAD_REVIEWER
 → DECIDE                (controller/human; no model)
 → RESUME                → BUILD | IDLE
```

Each phase has a fixed resident set (`PHASE_C_RESIDENTS`). Illegal skips raise
`ResidencyRefusal`. Reviews are **not** queued in C; they run in the REVIEW phase.

**When to use:** tight memory; long campaigns; reproducible sealed phase boundaries.

---

## 4. Mode transition graph

Any mode may transition to any mode (including itself as no-op), **subject to fit**:

```text
A ⇄ B ⇄ C ⇄ A
```

Laws:

1. Entering a mode with required slots checks `FitReport`.  
2. Mode switch clears loaded slots, then applies mode entry loads.  
3. Fit updates do **not** auto-degrade mode (silent residency change is a campaign hazard).  
4. Leaving B does not drop the review queue (pending reviews remain).

---

## 5. Interface (scaffold)

```text
lab/hcli/residency.py
  ResidencyMode, Slot, PhaseC
  FitReport, ReviewQueueItem, PipelinePair
  ResidencyState, ResidencyStateMachine
  ResidencyRefusal
  MODE_TRANSITIONS, PHASE_C_TRANSITIONS, PHASE_C_RESIDENTS
```

| Method | Mode | Purpose |
|--------|------|---------|
| `switch_mode` | all | fit-gated mode change |
| `update_fit` / `mode_fits` | all | host envelope |
| `load` / `unload` | all (A unload restricted) | logical slots |
| `pipeline_assign` | A | N / N+1 pairing |
| `enqueue_review` / `begin_review_drain` / `complete_review` | B | review queue |
| `advance_phase_c` / `run_phase_c_cycle` | C | phase chain |
| `snapshot` | all | sealed state receipt |
| `transition_tables` | static | export graph for docs/tests |

Schema: `hawking.hcli.residency.v1`.

---

## 6. Coupling to Option-C

| Option-C phase | Mode A | Mode B | Mode C |
|----------------|--------|--------|--------|
| EXECUTING | 30B (+ target) live; may pipeline | 30B+target live | BUILD phase |
| REVIEWING | 80B live in parallel | queued → drain window | REVIEW phase |
| CONTROLLER_EVAL | always; models optional | always | DECIDE phase |

Option-C remains logical: a candidate can complete its full phase machine under
any residency mode.

---

## 7. Phase plan (future)

### RM.0 — Scaffold (this revision)

- [x] Mode A/B/C state machine  
- [x] Fit gating  
- [x] Exhaustive Mode C edge walk in tests  
- [x] This plan  

### RM.1 — FitReport from live measurements

- Bind to free RAM / unified memory / Gravity working-set accounting.  
- Never decide fit from parameter count alone (same law as storage_modes).

### RM.2 — Runtime load/unload adapters

- Map `Slot` → actual model process / Metal residency handles.  
- Emit load/unload receipts; crash recovery returns to last sealed phase (Mode C).

### RM.3 — Pressure governor integration (Bible §27)

- YELLOW/RED may recommend or require B or C.  
- CRITICAL forbids Mode A entry.  
- Still no silent auto-switch without a ledger row.

### RM.4 — Option-C scheduler

- Mode A pipeline scheduler for candidate ids.  
- Mode B queue depth limits.  
- Mode C campaign driver calling `advance_phase_c` with sealed evidence at each step.

---

## 8. Non-negotiables

- Campaign continuity without permanent residency.  
- No silent mode degradation on fit change.  
- Mode A: no partial unload without mode switch.  
- Mode B: no reviewer co-resident with target.  
- Mode C: no phase skips; seal before unload where the chain requires it.  
- No real weight loading in this scaffold.  
- Do not bind to frankenstein live processes.

---

## 9. Exit criteria for "residency live"

1. RM.1–RM.2 complete with sealed load/unload receipts.  
2. One real campaign exercised in each of A, B, C.  
3. Tests still prove transition totality after runtime adapters land.  
4. Pressure governor cannot force an illegal phase skip.
