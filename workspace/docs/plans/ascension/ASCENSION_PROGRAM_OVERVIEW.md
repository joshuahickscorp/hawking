# Hawking Ascension — programme overview

**Authority:** `HAWKING_ASCENSION_BIBLE.md`  
**Primary product surface:** HCLI  
**Primary production backend:** Apple silicon / Metal  
**Bootstrap models:** Qwen3-Coder-30B (executor), Qwen3-Coder-Next 80B (reviewer)  
**Frankenstein:** excluded from this slice; returns later as a separate DeepSeek-family flagship  
**This document:** top-level index for the parallel ascension planning lanes  
**Durable trackers:**

| Tracker | Path |
|---------|------|
| Master schedule (steps 0–33) | [`ASCENSION_MASTER_SCHEDULE.json`](./ASCENSION_MASTER_SCHEDULE.json) |
| Completion state machine (12 states) | [`ASCENSION_COMPLETION_STATES.json`](./ASCENSION_COMPLETION_STATES.json) |
| HCLI product-test plan | [`ASCENSION_HCLI_PRODUCT_TEST_PLAN.md`](./ASCENSION_HCLI_PRODUCT_TEST_PLAN.md) |
| HCLI product-test catalog | [`ASCENSION_HCLI_PRODUCT_TEST_CATALOG.json`](./ASCENSION_HCLI_PRODUCT_TEST_CATALOG.json) |

All schedule steps start **`NOT_STARTED`**. All completion states start **`CANDIDATE`**.  
Sandbox models may report candidates; only controller/human certifies terminal states (bible §36).

---

## Final directive (compressed)

> Research the physical path → encode it into Gravity → make the Qwen pair fast enough to build the laboratory → give HCLI memory, search, skills, perception and communication → let **Hawking**, not sandbox models, decide what becomes infrastructure.

Success loop:

```text
find a paper or mechanism
→ retrieve evidence
→ bounded experiment
→ implement
→ verify
→ challenge independently
→ record
→ clean up
→ resume
→ rotate
```

TG3 is the mandatory review gate. TG2/TG1 remain the frontier.

---

## Bible section → plan index

Paths are under `workspace/docs/plans/ascension/` unless noted.  
**pending companion lane** means the owning parallel lane has not landed the file in this worktree yet — do not invent its content.

### Platform, authority, physics (bible §0–3)

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 0 | Canonical project definition | this overview + master schedule | ascension-master-schedule |
| 1 | Platform decision (Apple Tier 1; CUDA future Tier 1B) | `ASCENSION_PLATFORM_DECISION_PLAN.md`; `ASCENSION_PLATFORM_DECISION_CONTRACT.md` | ascension-platform-decision |
| 2 | Human and controller authority | completion states + governance plans | ascension-governance-scaffold |
| 3 | Governing physical model | `ASCENSION_GRAVITY_RESEARCH_REGISTRY.md` | ascension-gravity-research — **pending companion lane** |

### Research before construction (bible §4–5, §32)

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 4 | Research before full artifact construction | `ASCENSION_GRAVITY_RESEARCH_REGISTRY.md` | ascension-gravity-research — **pending companion lane** |
| 5 | Research portfolio (B/F/U/R/D/S/K mechanisms) | `ASCENSION_GRAVITY_RESEARCH_REGISTRY.md`, `ASCENSION_RESEARCH_REGISTRY_PLAN.md` | gravity-research + acquisition-registry — **pending companion lane** |
| 32 | Negative-science inheritance | `ASCENSION_GRAVEYARD_PLAN.md`, research registry | ascension-acquisition-registry — **pending companion lane** |

### Gravity gate, acquisition, credentials (bible §6–7)

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 6 | Gravity co-design gate | `ASCENSION_GRAVITY_RESEARCH_REGISTRY.md` | ascension-gravity-research — **pending companion lane** |
| 7 | Source acquisition and credential broker | `ASCENSION_CREDENTIAL_BROKER_PLAN.md`, `ASCENSION_RESEARCH_REGISTRY_PLAN.md`, `ASCENSION_GRAVEYARD_PLAN.md` | ascension-acquisition-registry — **pending companion lane** |

### Dual-Qwen bootstrap, TG, profiler (bible §8–11)

`ASCENSION_BLOCKED_STATE_REGISTRY.json` is the canonical machine-readable
download gate for both bootstrap families. Its current `BLOCKED` state is
intentional: planning documents do not authorize a model-body acquisition.

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 8 | Bootstrap model 1 — Qwen3-Coder-30B | `ASCENSION_30B_PARITY_LADDER_PLAN.md`; blocked-state registry | ascension-bootstrap-parity — **pending companion lane** |
| 9 | Bootstrap model 2 — Qwen3-Coder-Next 80B | `ASCENSION_80B_HYBRID_ARCHITECTURE_PLAN.md`; blocked-state registry | ascension-bootstrap-parity — **pending companion lane** |
| 10 | Self-TG gauntlets before sandbox admission | `ASCENSION_TG_GAUNTLET_HARNESS_PLAN.md` | ascension-bootstrap-parity — **pending companion lane** |
| 11 | Complete-token profiler and FLOPS ledger | `ASCENSION_COMPLETE_TOKEN_PROFILER_PLAN.md`; `ASCENSION_COMPLETE_TOKEN_PROFILER_CONTRACT.md` (+ TG / 30B / 80B plans) | ascension-profiler-ledger |

### HCLI Agent OS (bible §12–22)

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 12 | Agent OS overview | this overview; schedule step 9 | hcli-scheduler-planning + siblings |
| 13 | Agent scheduler | `HCLI_AGENT_SCHEDULER_PLAN.md` | hcli-scheduler-planning — **pending companion lane** |
| 14 | Planning diagnostic | `HCLI_PLANNING_DIAGNOSTIC_PLAN.md` | hcli-scheduler-planning — **pending companion lane** |
| 15 | Agent retrieval gateway | `HCLI_RETRIEVAL_GATEWAY_PLAN.md` | hcli-retrieval-tools — **pending companion lane** |
| 16 | Tool gateway | `HCLI_TOOL_GATEWAY_PLAN.md` | hcli-retrieval-tools — **pending companion lane** |
| 17 | Memory OS | `HCLI_MEMORY_OS_PLAN.md` | hcli-memory-skills — **pending companion lane** |
| 18 | Skill Foundry | `HCLI_SKILL_FOUNDRY_PLAN.md` | hcli-memory-skills — **pending companion lane** |
| 19 | Perception service | `HCLI_PERCEPTION_SERVICE_PLAN.md` | hcli-perception-comms — **pending companion lane** |
| 20 | Communication bus | `HCLI_COMMUNICATION_BUS_PLAN.md` | hcli-perception-comms — **pending companion lane** |
| 21 | Execution sandbox | `HCLI_EXECUTION_SANDBOX_PLAN.md` | hcli-sandbox-verification — **pending companion lane** |
| 22 | Verification authority | `HCLI_VERIFICATION_AUTHORITY_PLAN.md` | hcli-sandbox-verification — **pending companion lane** |

### Evolution, Option-C, residency (bible §23–25)

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 23 | Self-evolution engine | `HCLI_SELF_EVOLUTION_PLAN.md` | hcli-evolution-optionc — **pending companion lane** |
| 24 | Option-C sandbox | `HCLI_OPTION_C_PLAN.md` | hcli-evolution-optionc — **pending companion lane** |
| 25 | Residency modes (A/B/C) | `HCLI_RESIDENCY_MODES_PLAN.md` | hcli-evolution-optionc — **pending companion lane** |

### Garbage, pressure, notifications (bible §26–28)

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 26 | Garbage ecosystem | `ASCENSION_GARBAGE_ECOSYSTEM_PLAN.md` | ascension-governance-scaffold — **pending companion lane** |
| 27 | Pressure governor | `ASCENSION_PRESSURE_GOVERNOR_PLAN.md` | ascension-governance-scaffold — **pending companion lane** |
| 28 | Notifications | `ASCENSION_NOTIFICATIONS_PLAN.md` | ascension-governance-scaffold — **pending companion lane** |

### Family ladder and rotation (bible §29–31)

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 29 | Family kernel architecture | `ASCENSION_FAMILY_KERNEL_ARCHITECTURE_PLAN.md`; `ASCENSION_FAMILY_KERNEL_LADDER_ROTATION_CONTRACT.md` | ascension-family-rotation |
| 30 | Wider autonomous model ladder | `ASCENSION_MODEL_LADDER_PLAN.md`; `ASCENSION_FAMILY_KERNEL_LADDER_ROTATION_CONTRACT.md` | ascension-family-rotation |
| 31 | Rotation rule | `ASCENSION_ROTATION_RULE_PLAN.md`; `ASCENSION_FAMILY_KERNEL_LADDER_ROTATION_CONTRACT.md` | ascension-family-rotation |

### Product tests, CUDA, schedule, completion (bible §33–36) — this lane

| § | Topic | Plan / tracker | Owner lane |
|---|-------|----------------|------------|
| 33 | HCLI product tests | `ASCENSION_HCLI_PRODUCT_TEST_PLAN.md`, `ASCENSION_HCLI_PRODUCT_TEST_CATALOG.json`, harness under `tools/condense/hcli_product_test_harness.py` | **ascension-master-schedule** |
| 34 | CUDA future (deferred, not rejected) | this overview §CUDA; master schedule step 33; preserve list in schedule JSON | **ascension-master-schedule** |
| 35 | Canonical execution schedule | `ASCENSION_MASTER_SCHEDULE.json` | **ascension-master-schedule** |
| 36 | Completion states | `ASCENSION_COMPLETION_STATES.json` | **ascension-master-schedule** |

---

## Parallel lane map (10 planning lanes)

| Lane id | Bible focus | Expected deliverables under `workspace/docs/plans/ascension/` |
|---------|-------------|----------------------------------------------------------------|
| `ascension-master-schedule` | §33–36 + Final Directive | this overview, master schedule JSON, completion states JSON, HCLI product-test plan/catalog/harness |
| `ascension-gravity-research` | §3–6, research portfolio | `ASCENSION_GRAVITY_RESEARCH_REGISTRY.md` — **pending companion lane** |
| `ascension-acquisition-registry` | §4–5, §7, §32 | `ASCENSION_CREDENTIAL_BROKER_PLAN.md`, `ASCENSION_RESEARCH_REGISTRY_PLAN.md`, `ASCENSION_GRAVEYARD_PLAN.md` — **pending companion lane** |
| `ascension-bootstrap-parity` | §8–11 | `ASCENSION_30B_PARITY_LADDER_PLAN.md`, `ASCENSION_80B_HYBRID_ARCHITECTURE_PLAN.md`, `ASCENSION_TG_GAUNTLET_HARNESS_PLAN.md` — **pending companion lane** |
| `ascension-governance-scaffold` | §26–28 | `ASCENSION_GARBAGE_ECOSYSTEM_PLAN.md`, `ASCENSION_PRESSURE_GOVERNOR_PLAN.md`, `ASCENSION_NOTIFICATIONS_PLAN.md` — **pending companion lane** |
| `hcli-scheduler-planning` | §12–14 | `HCLI_AGENT_SCHEDULER_PLAN.md`, `HCLI_PLANNING_DIAGNOSTIC_PLAN.md` — **pending companion lane** |
| `hcli-retrieval-tools` | §15–16 | `HCLI_RETRIEVAL_GATEWAY_PLAN.md`, `HCLI_TOOL_GATEWAY_PLAN.md` — **pending companion lane** |
| `hcli-memory-skills` | §17–18 | `HCLI_MEMORY_OS_PLAN.md`, `HCLI_SKILL_FOUNDRY_PLAN.md` — **pending companion lane** |
| `hcli-perception-comms` | §19–20 | `HCLI_PERCEPTION_SERVICE_PLAN.md`, `HCLI_COMMUNICATION_BUS_PLAN.md` — **pending companion lane** |
| `hcli-sandbox-verification` | §21–22 | `HCLI_EXECUTION_SANDBOX_PLAN.md`, `HCLI_VERIFICATION_AUTHORITY_PLAN.md` — **pending companion lane** |
| `hcli-evolution-optionc` | §23–25 | `HCLI_SELF_EVOLUTION_PLAN.md`, `HCLI_OPTION_C_PLAN.md`, `HCLI_RESIDENCY_MODES_PLAN.md` — **pending companion lane** |

> Note: the task brief called out “9 parallel lanes” beside this one; the live wave also includes the six HCLI Agent OS lanes above (11 named workstreams total including master-schedule). All are indexed here so the programme has one map.

---

## Canonical schedule at a glance (bible §35)

See `ASCENSION_MASTER_SCHEDULE.json` for prerequisites, companion docs, and status.

```text
 0 Proto-Frankenstein offload
 1 Freeze authorities / binaries
 2 Report-only model authority
 3 Garbage ownership + pressure governor
 4 Credential broker
 5 Research registry + Graveyard
 6 Audit both Qwen architectures
 7 Prototype B/F/U/R/D/S/K mechanisms
 8 Classify mechanisms
 9 HCLI Agent OS foundations
10 Gravity direct-execution layouts
11 Stream + Gravity 30B
12 Exact-model 30B path
13 Parity / capability / profiler
14 30B self-TG
15 TG3 review (30B)
16 Promote 30B executor
17 Stream + Gravity 80B
18 Exact-model 80B hybrid path
19 Port applicable 30B wins
20 80B parity / capability / profiler
21 80B self-TG
22 TG3 review (80B)
23 Promote 80B reviewer
24 Option-C sandbox
25 Residency mode tests
26 Production retrieval / memory / skills / perception
27 Isolated latent communication experiments
28 Qwen family graphs
29 Autonomous wider model ladder
30 Rotate harder families
31 TG2/TG1 research under human/controller authority
32 Seal Apple production release
33 CUDA as separately funded post-release backend
```

---

## Completion states at a glance (bible §36)

See `ASCENSION_COMPLETION_STATES.json` for the full gate graph.

```text
PROTO_FRANKENSTEIN_OFFLOADED
HAWKING_RESEARCH_PORTFOLIO_FROZEN
HCLI_AGENT_OS_FOUNDATION_READY
QWEN3_30B_GRAVITY_READY
QWEN3_30B_EXECUTOR_READY
QWEN3_NEXT_80B_GRAVITY_READY
QWEN3_NEXT_80B_REVIEWER_READY
TG3_REVIEW_REQUIRED                 ← re-entrant review gate
HAWKING_OPTION_C_KERNEL_SANDBOX_READY
HCLI_AGENT_PIPELINE_READY
HAWKING_SELF_CONTAINED_MODEL_LADDER_ACTIVE
HAWKING_APPLE_PRODUCTION_RELEASE_READY   ← terminal Apple slice
```

---

## CUDA future (bible §34)

**Deferred, not rejected.** After Apple production release (step 32), step 33 may begin only as a separately funded backend.

Preserve backend-neutral:

```text
architecture IR
Gravity tensor semantics
kernel grammar
benchmark contracts
parity/capability suites
receipt schema
scheduler API
```

Later CUDA tools may include CUDA Graphs, CUTLASS/cuBLAS, custom CUDA/Triton kernels, async pipelines, device-specific autotuning — with real CUDA hardware and independent parity/performance evidence. **Apple claims remain Apple-specific.**

---

## HCLI product tests (bible §33)

Primary metric: **verified tasks completed / hour**.  
Harness generalizes the existing DeepSeek diagnostic live suite (`test_deepseek_v4_hcli_live_suite.py`, `hcli_live_suite_receipt`, evidence under `workspace/campaign/evidence/hide/deepseek-v4-live.*`).

Case list: chat, repo context, coding, planner/act/verify, tool calls, structured JSON, session restart, endpoint restart, context compaction, read-safe swarm, isolated write-agent, continuous batching, search/retrieval, memory operations, skill execution, document perception.

Details: [`ASCENSION_HCLI_PRODUCT_TEST_PLAN.md`](./ASCENSION_HCLI_PRODUCT_TEST_PLAN.md).

---

## Related pre-existing plans (not ascension-named, still relevant)

These predate the ascension wave and must not be confused with completion of ascension steps:

| Path | Relevance |
|------|-----------|
| `workspace/docs/plans/POST_FINAL_GRAVITY_HCLI_PLAN.md` | Post-Final Frankenstein Gravity + HCLI Terra/Luna path — separate DeepSeek-family track |
| `workspace/docs/plans/FRANKENSTEIN_PROGRAM.md` | Frankenstein programme — **excluded** from this slice |
| `workspace/docs/plans/PROTO_FRANKENSTEIN_V0_STEER.md` | Proto-Frankenstein — schedule step 0 must offload before active work |
| `workspace/docs/plans/STAGE2_KIMI_STREAMING_DISTILL_PLAN.md` | Stage-2 distill — not an ascension gate |
| `workspace/docs/plans/KERNEL_BROKERS_TUNING_PLAN.md` | Kernel tuning — informs Gravity research |

---

## How to advance the programme (operators)

1. Read this overview for section ownership.
2. Open `ASCENSION_MASTER_SCHEDULE.json` for the next `NOT_STARTED` step whose prerequisites are certified.
3. Open the companion plan for that step (or wait if still **pending companion lane**).
4. Produce sealed evidence; sandbox may set `SANDBOX_REPORTED` / `CANDIDATE_COMPLETE` only.
5. Controller/human sets `CONTROLLER_CERTIFIED` on the completion state and step.
6. Never mark Apple production or TG promotions complete from a sandbox model report alone.

---

## Non-goals of this planning wave

- No live Qwen training/serve work in planning lanes  
- No CUDA implementation  
- No mutation of `lab/operators/frankenstein_*` or live frankenstein evidence during GLM recapture  
- No remote push/PR as part of planning scaffolds  
- No silent status flips to “done”
