# aud10 — HCLI boundary (audit only)

HEAD `04193ccbc`. H-ROADMAP at `/Users/scammermike/Downloads/H-ROADMAP.md`. This lane did not edit `hcli/`, did not signal the live daemon, and did not rewrite the roadmap.

Evidence class is **STATIC_VERIFICATION** plus **SOURCE_INSPECTION** of live disk. Nothing in this file is `PHYSICALLY_MEASURED`. Process PIDs are **BLOCKED_AUTHORITY** in this sandbox (`ps` denied; `launchctl list` empty).

## Answer

HCLI already owns a real control plane on HEAD: Goal Compiler, hawkingd, watch, context compiler, WorkUnitDAG, MAXX scheduler, escalation tools, Odyssey facade, EventSink, and evidence-child refill. It does **not** own the named `WorkGraph` type, ModelLake specimen-arrival ingestion, or self-mutation on the agent loop. Mixed MAX and unattended metabolism are not proven. The live resident body is **UNLOADED** and the live mission is **cancelled**.

The roadmap still talks as if I-A were a list of genes-to-express and a 1-hour/712-test snapshot. That wording is what should change — not the five-era constitution.

| Capability | State | Production call (not import) | Blocker |
|---|---|---|---|
| Goal Compiler | **INTEGRATED** | `GoalCompiler.compile` `hcli/engine.py:1292`, `hcli/mission.py:316`, `hcli/controller.py:1399` | Live stored IR dropped obligations; `accepted_count=0` |
| self-hosted verifier synthesis | **INTEGRATED** | `GoalCompiler._synthesised_verify_command` `hcli/goal.py:358`; `run_pipeline` `hcli/delegate.py:1333`; `command_is_admissible` `hcli/ledger.py:689` | G001 canaries non-production paths (`atomic_write`, `hcli/resident.py`, `tools/reach.py`) |
| resident daemon | **INTEGRATED** | `daemon_main` `hcli/hawkingd.py:31`; `start_resident` `hcli/agentos_cli.py:1233` | Body UNLOADED; hawkingd PID unconfirmed; genesis launchd is a different process |
| watch/control plane | **INTEGRATED** | `watch_resident` `hcli/agentos/resident.py:2281` | `G015_control_plane.json` **ABSENT** |
| self-mutation | **CALLABLE** | `apply_mutation_operations` `tools/future/resident_code_tools.py:552` | **No** `hcli/engine.py` / `tool_registry.py` caller; `G003` **ABSENT** |
| successor handoff | **CALLABLE** | `retire_incumbent` `hcli/agentos/resident.py:2050` (replace path) | `G013_successor.json` **ABSENT** |
| context compiler | **END_TO_END** | `compile_worker_context` `hcli/mission.py:1055` | Sovereign `G004` **ABSENT**; acceptance receipts are FUNCTIONAL_SIM |
| WorkGraphs | **INTEGRATED** as WorkUnitDAG | `WorkUnitDAG` `hcli/goal_compile.py:268`; `DagStore` `hcli/scheduler.py:107` | Named `WorkGraph` is `tools/future/workgraph.py` (`executes=false`) |
| MAXX / global scheduling | **INTEGRATED** | `Scheduler` `hcli/mission.py:264`; `grok_pool_snapshot` `hcli/controller.py:337` | `HCLI_MIXED_MAX` **BLOCKED** (no cognition backend) |
| provider escalation | **TESTED** | `escalate_to_frontier` `hcli/tool_registry.py:1516` | Live compact catalog omits `grok.swarm.*` |
| ModelLake event ingestion | **CALLABLE** | `EventSink.write` `hcli/agentos/resident.py:1585` (control-plane, not lake); `odyssey.ingest` registered | No specimen-arrival → HCLI DAG caller inside `hcli/` |
| Odyssey ownership | **INTEGRATED** as facade | G009 invoked `odyssey.status` and `odyssey.queue` | Driver is still `tools/odyssey`; `record_law`/`record_scar` test-only; II/III **BLOCKED** |
| autonomous scientific metabolism | **CALLABLE** | `admit_evidence_children` `hcli/agentos/resident.py:1012` | `G008` **ABSENT**; live `accepted_count=0`; `HCLI_SELF_OPTIMIZATION_BOOTSTRAP` **BLOCKED** |

`goal_bank.py`, `goal_compile.py`, and `knowledge.py` are **committed at HEAD**, not uncommitted. This worktree never materialized `hcli/`; source was read from git blobs.

## What HCLI now owns (and what it does not)

**Owns, with callers of the symbol itself:**

- Goal Compiler → DAG, synthesised verify commands, worker packets, goal bank
- hawkingd supervisor/worker split, CLI `hcli resident …`, shim `hawkingd`
- read-only watch plane that tails `.hcli/mission/events.jsonl` and must not open a second body
- context compiler on Mission dispatch (acceptance ACCEPTED 2026-09-02)
- WorkUnitDAG + GoalGraph + DagStore (restart/repair/schedule)
- Scheduler + `max_policy.resolve_grok_admitted` + status snapshot
- provider escalation tools, fail-closed without credentials
- Odyssey `status`/`queue`/`ingest`/`cycle` tools (confirm-gated)
- EventSink durable resident log
- `admit_evidence_children` (self-supplement ACCEPTED)

**Does not own, or is not on the HCLI loop:**

- `class WorkGraph` in `tools/future/workgraph.py` (STATIC_ONLY schedule emitter; `execute()` refused)
- `apply_mutation_operations` on the agent loop (lives in `hcli/mutation.py`, called from `tools/future`)
- ModelLake specimen arrival as an HCLI event
- Odyssey driver, Odyssey II transfer, Odyssey III adversarial science
- mixed MAX with a live model backend
- unattended multi-replan metabolism
- a currently loaded resident body

## Live disk (read-only, Downloads/hawking)

Mission `8ee9a7d3-64f6-4fc5-a53f-66e30f9a959f`: phase `cancelled`, reason `resident_self_evacuation`, 79 units (34 pending / 11 failed / 34 ready), `accepted_count=0`. Evidence rows say `NO_EVIDENCE`. Stored `compiled` has summary/invariants/acceptance only — no obligations. Sampled G001–G005 units all verify with `python3 -m pytest hcli/test_goal_verifier_synthesis.py` because the live goal named those files.

Resident `body.json`: `status=UNLOADED`, `unload_reason=supervisor_stopped`, `worker_pid=null`, behavior `RESTART_WORKER`.

`com.hawking.genesis` launchd runs `tools/genesis_forever.sh` → `tools/agentos/genesis_resident.py`. That is not `python -m hcli.hawkingd`.

## Tests this session

Extracted HEAD python, `PYTHONPATH` set, `python3 -m pytest … -o addopts=""`:

- 151 passed (goal compile/graph/ir/bank/tokenizer, event sink, frontier scheduler, hawkingd name, command registry, escalation)
- 39 passed (goal-compiler acceptance + source compile)
- Odyssey: 18 confirm-gate tests passed; 3 live-state tests failed because the extract tree is not the live Odyssey workspace

Protected G-gates that only load receipts: **10 of 13 receipts absent** (`G002 G003 G004 G005 G006 G007 G008 G012 G013 G015`). Present: `G001 G009 G010 G014`. A receipt-gated test is not a passing measurement.

## Surprises (loud)

1. Contract said ~110 uncommitted `hcli/` files. They are in HEAD. Live dirty set was three files, then empty. Another session is finishing HCLI.
2. `civilization/CAPABILITY_GRAPH.json` marks I-A **SCAFFOLDED** with `runtime_caller=[]` off commit `7d64280`. That is the wrong grain. `GoalCompiler().compile` has production callers on HEAD.
3. G001 “completed” against synthetic `atomic_write` / `hcli/resident.py::SwapoutsProbe` / `tools/reach.py`. Production is `atomic_write_text/json`, `hcli/agentos/resident.py`, `tools/roadmap/reach.py`. No `SwapoutsProbe` found.
4. Named WorkGraph is not HCLI. HCLI’s graph is WorkUnitDAG.
5. Live body UNLOADED; genesis launchd ≠ hawkingd.
6. Self-mutation primitive has zero `hcli/` production apply-callers.

What would settle each surprise is in the JSON `surprises[].what_would_settle_it`.

## Roadmap wording that should change

Do not add Era VI, a fourth Odyssey, or a Theia civilization. Do not casually rewrite 0.7%. Change inventory and tense:

1. **§3 snapshot (H-ROADMAP.md:351–365)** — drop frozen 712 tests / 3600 s / A1–A6 / 144-file Flash tree as if they were live. Point at disk: `ROADMAP_STATE.json`, `.hcli/mission/state.json`, `.hcli/resident/body.json`, `receipts/acceptance/HCLI_*.json`.
2. **§5 I-A genes (428–445) and Appendix A subgenes (5296–5307)** — add the organs that now have callers (Goal Compiler, hawkingd, watch, context compiler, WorkUnitDAG, MAXX scheduler, escalation, Odyssey facade, EventSink) with the states above. Keep homeostasis genes. Fill the gene-card AUTHORITY slots with those artifacts instead of the generic CRISPR template.
3. **§7 diagram (1028–1052)** — client vs hawkingd vs disposable body; Goal Compiler before the DAG; watch as a read-only sibling that must not open a second body; context compiler on the worker packet.
4. **§7.4 / §7.5 (1109–1142)** — Scheduler exists; mixed MAX is blocked; live disk is not L3. Do not let the autonomy ladder read as present-tense L3/L4.
5. **Pocket card HCLI line (5138–5141)** — add GoalCompiler, hawkingd, context compiler, WorkUnitDAG; say the named WorkGraph is not this organ.
6. **D.13 ultragoal/bootstrap/compaction (7646–7684)** — August 23 `receipts/headless/HCLI_ULTRAGOAL_INGRESS.json` etc. have no schema/verdict. Mark NOT_RUN/INCONCLUSIVE until a `hawking.acceptance.gate.v1` receipt exists.
7. **`civilization/CAPABILITY_GRAPH.json` I-A SCAFFOLDED** — regenerate on HEAD with capability-level symbols.
8. **`civilization/ROADMAP_STATE.json` `active_odyssey=ADVERSARIALLY_VERIFIED`** — do not let that be read as HCLI owning Odyssey. Odyssey II/III acceptance is BLOCKED.
9. **Unqualified “WorkGraph” (≈1959)** — HCLI WorkUnitDAG vs `tools/future` WorkGraph (`executes=false`).
10. **`hcli/test_odyssey.py` docstring** — “HCLI has no odyssey verb at all” is false; `odyssey.*` tools exist and G009 invoked two.

## Unlocks

- Goal Compiler unlocks verifier commands, worker packets, DAG, ultragoal ingress.
- hawkingd unlocks watch, successor replace, EventSink, evidence children.
- Context compiler unlocks focused WorkUnits and MAX admission.
- WorkUnitDAG unlocks Scheduler, repair, restart.
- Verifier synthesis unlocks promotion that is not self-certification — once synthesised commands name production symbols.
- Self-mutation, mixed MAX, G008 metabolism, G013 successor, and ModelLake→DAG admission do **not** unlock anything further until their blockers move.

## What this lane did not do

No `H-ROADMAP.md` rewrite. No implementation campaign. No `hcli/` edit. No process signal. No `MEASURED` claim.
