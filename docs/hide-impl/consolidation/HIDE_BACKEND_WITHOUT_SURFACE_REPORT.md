# HIDE Backend Without Surface Report

Consolidation census, writer B. THE key gap file, and the driver for the consolidation campaign.
Every host capability, protocol method, and intent that is built, durable, and tested but has NO
frontend path, grouped by domain. Each entry names the EXISTING control it should be exposed under
(never a new button) and a one-line productivity rationale. Grounded in the merged census catalog
(branch `build/hide-impl-2026-07-19`, read-only). UNKNOWN where the catalog is silent.

House note: hyphens and parentheses only, no long dashes.

---

## The headline

The shipped FE speaks exactly three routes (`app/src/ipc.ts`): `POST /v1/hide/intent` (Wire-A
typed intents), `WS|GET /v1/hide/events` (Wire-B), `POST /v1/hide/connector`. It NEVER posts to
`POST /v1/hide/rpc`. So the entire elevated Agent-Server protocol surface is server-wired and typed
but FE-dark: the census confirms ZERO of the 48 RPC Methods is reachable from `app/src`.

That produces three distinct backend-without-surface classes, all covered below:

1. Whole host domains built + durable + tested with NO route at all (fe_reachable = `no`).
2. Capabilities reachable in principle over `/rpc` OR via a `Custom{name}` intent, but FE-dark
   because the FE never dials `/rpc` and 8 host-handled custom names are absent from `wire.ts`.
3. A genuine end-to-end hole (turn/steer) where protocol, item, and InterruptHub variant all exist
   but nothing ever signals it.

The lazy consolidation fix for class 2 is almost always: add the missing custom name to
`wire.ts` CUSTOM_NAMES and route an EXISTING control to it. Do not teach the FE a second `/rpc`
client, and do not add buttons; the surfaces below already exist.

---

## Class 1: Whole domains built with NO route (fe_reachable = no)

### Memory domain (KV `memory`, outcome-governed; host.rs:1870-1997)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `memory_add` | `host.rs:1870` | ContextStack "inject memory note" (`ContextStack.tsx:245`) | Durable note the agent actually keeps, instead of the current LOG-ONLY `pin_span`. |
| `memory_supersede` | `host.rs:1906` | ContextStack "inject note" edit + "pin dropped span" (`ContextStack.tsx:268`) | Replace a stale fact without losing its history (writes Active + Superseded). |
| `memory_record_outcome` | `host.rs:1931` | ContextStack "evict memory" (`ContextStack.tsx:231`) | Report that a remembered fact was wrong so it self-quarantines. |
| `memory_revalidate` | `host.rs:1957` | ContextStack Memory stratum header | Re-check a memory's citations against the repo on disk before reuse. |
| `memory_context` | `host.rs:1892` | ContextStack Memory stratum (read view) | Show the exact context-eligible subset the compiler draws from. |
| `memory_get` / `memory_list` | `host.rs:1877` / `1885` | ContextStack Memory stratum list | Inspect what the agent remembers (today rendered from mock/seed). |

One-line domain rationale: durable outcome-governed memory stops the agent re-deriving the same
repo facts every session; the Memory stratum already renders the data and its pin/evict/note
controls fire LOG-ONLY `pin_span`/`unpin_span`, they just need to point at these writes.

### Job domain (KV `jobs` + job.created/status/cancelled events; host.rs:1743-1846)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `job_create` | `host.rs:1743` | HomeComposer send path / Home stage "Tools" panel tab (`Home.tsx:170`) | Queue a durable background job that survives restart. |
| `job_list` / `job_get` | `host.rs:1763` / `1758` | Home Digest activity view (`Digest.tsx`) or the "Tools" panel | See pending/running jobs where activity is already shown. |
| `job_update_status` | `host.rs:1784` | (surfaced read-only via job_list; status is host-driven) | Reflect real job state in the same panel. |
| `job_cancel` | `host.rs:1818` | reuse the FleetView "stop" pattern (`FleetView.tsx:72`) | Cancel a job with the same gesture that stops a fleet attempt. |
| `job_evaluate_triggers` / `jobs_recover` | `host.rs:1775` / `1846` | (internal predicate / boot rebuild) | No direct control needed; drives wake + startup recovery behind the list. |

One-line domain rationale: durable background jobs let long tasks outlive the session and wake on
triggers (execution is DEFERRED_MODEL_REQUIRED, but create/list/cancel governance is ready now).

### Workspace domain (multi-repo trust graph, KV; host.rs:1122-1182)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `workspace_add_repo` | `host.rs:1122` | HomeComposer "Add folder" (`HomeComposer.tsx:245`, sends `open_folder`) | The add-folder flow is the one place a repo enters the workspace. |
| `workspace_set_repo_trust` | `host.rs:1136` | a trust prompt inside the same Add-folder flow | CRITICAL: repos enter UNTRUSTED by default; with no trust surface their instructions/policy stay inert with no activation path. |
| `workspace_repo` / `workspace_graph` | `host.rs:1128` / `1182` | SideBar model/context popover (`SideBar.tsx:53`) or Settings | Read the multi-repo graph where model/context is already shown. |
| `workspace_add_environment` / `workspace_environment` / `workspace_add_edge` | `host.rs:1154` / `1160` / `1167` | Settings workspace section | Compose the repo+environment graph from the existing settings surface. |

One-line domain rationale: without `set_repo_trust` a repo's CLAUDE.md and policy can never activate;
the Add-folder flow already exists and is the correct home for the trust decision.

### Environment domain (environment.switch event + current_env pointer; host.rs:1193-1233)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `environment_switch` | `host.rs:1193` | SideBar model/context popover (`SideBar.tsx:53`) or a chip beside HomeComposer Effort/Model | Switch dev/prod/sandbox per session without restart. |
| `environment_switches` | `host.rs:1233` | same popover (history read) | Show which environment is active; the `environment_switch` UiEvent is already emitted. |

### Verify domain (deterministic verification plane; host.rs:1373-1479)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `run_static_analysis` | `host.rs:1373` | StatusBar Problems counter (`StatusBar.tsx:31`, currently a hardcoded 0/0 mock) | Turn the fake 0/0 into real Tier-1 static-analysis counts. |
| `verification_receipts` | `host.rs:1434` | ContextStack "Tests & state" stratum / the same Problems item | Show sealed pass/fail receipts the human can trust. |
| `reconcile_review_for_scope` | `host.rs:1479` | the diff-review / gate flow | Enforce that a deterministic Fail cannot be review-overridden before accept. |
| `review_role_profiles` / `review_role_profile` | `host.rs:1458` / `1465` | Settings or ContextStack Tools stratum (read-only data) | Expose which review roles exist (executing one is DEFERRED_MODEL_REQUIRED). |

One-line domain rationale: surface real deterministic verification so the user sees true problem
counts and cannot merge over a failing gate, instead of a hardcoded 0/0.

### Effect / policy domain (typed effect ledger; host.rs:1278-1339)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `evaluate_tool_policy` | `host.rs:1278` | Security Gate overlay (`App.tsx:305`) + InlineGate (`structure.tsx:300`) | The approve/deny surface already exists; show the policy decision that produced the gate. |
| `policy_decisions` | `host.rs:1339` | a history view in the same gate surface / ContextStack Tools stratum | Audit why each tool was Allowed / Ask / RequireSandbox / Denied. |

### Program domain (bounded programmatic-tool interpreter; program.rs:448-547)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `run_program` | `program.rs:448` | Terminal (`Terminal.tsx:150`) for invocation | Safe multi-step tool runs from the terminal the user already has. |
| `run_program_with_limits` | `program.rs:465` | same, with a limits affordance | Its WriteProposals should feed the EXISTING Editor/Home diff-review (`accept_diff`/`reject_diff`), so writes land in the review the user already gates. |

### Goal domain (KV `goals`; host.rs:1521-1589)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `goal_evaluate` | `host.rs:1572` | HomeComposer (goal+acceptance field beside the task textarea, `HomeComposer.tsx:209`) or PlanCard header | Deterministic acceptance check vs verify.result lets the agent self-verify completion. |
| `goal_set` / `goal_get` / `goal_clear` | `host.rs:1521` / `1539` / `1546` | same HomeComposer goal field | Set/read/clear a durable goal. NOTE: `goal_set`/`goal_clear` ARE host-handled but their custom names are absent from `wire.ts`, so the whole domain is FE-dark (see Class 2). |

One-line domain rationale: a durable goal plus a deterministic acceptance test lets the agent stop
exactly when done instead of halting mid-task or over-running.

### Session diagnostics (host.rs:2273-2284)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `status` | `host.rs:2273` | StatusBar (`shell/StatusBar.tsx`) | Real capabilities/connectors/tools/runtime snapshot instead of a static label. |
| `health` | `host.rs:2284` | Settings Engine section / StatusBar indicator | Structured health over layout/tools/connectors/runtime; `/healthz` returns only a static "ok" string (`hide-serve/lib.rs:90`). |

### Steering (the genuine end-to-end hole)

| Capability | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `turn/steer` (InterruptHub Steer variant) | `interrupt.rs:32`; `rpc.rs:374` returns NotImplemented | SteerBar Redirect/Steer (`chat/SteerBar.tsx:56` / `:70`) | The SteerBar UI is fully built and already fires `redirect_run`, but `redirect_run` is LOG-ONLY host-side and nothing signals InterruptHub Steer. Wire it so mid-turn redirect actually alters the running turn instead of only appending a log event. |

---

## Class 2: Built and protocol-wired but FE-dark

These have a real host capability behind a route the shipped FE never uses. Two sub-cases.

### 2a. Host-handled custom names ABSENT from wire.ts CUSTOM_NAMES

`intent.custom()` cannot send these (they fail the CustomName type), so the host paths are
unreachable from the typed FE even though they are fully handled.

| Custom name | host ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `create_side_chat` / `merge_side_chat` | `host.rs:413` / `427` | the DEAD "New chat" buttons in ChatPane (`ChatPane.tsx:79`) and FloatingChat (`FloatingChat.tsx:69`), plus a merge action in the pane header | The exact surface already exists as dead buttons; wiring costs a registry entry, not a new control. |
| `approve_effect` / `deny_effect` | `host.rs:397-403` | Security Gate overlay + InlineGate | Reuse the existing approval overlay for effect-level (paused kernel step) approvals, not just RunCommand gates. |
| `goal_set` / `goal_clear` | `host.rs:506` / `526` | HomeComposer goal field (see Goal domain) | Send/clear a durable goal from the composer. |
| `checkpoint_create` / `checkpoint_restore` | `host.rs:533` / `550` | StateTimeline (`shell/StateTimeline.tsx`, "fork from here"/scrub/"live" row) | Real integrity-verified restore points on the timeline the FE already renders; also fixes the mock ContextStack "snapshot state". |

### 2b. RPC Methods that dispatch to real host caps, but the FE never posts to /rpc

The lazy path is to reach these via a `/intent` custom name (add to `wire.ts`), not to add a
second `/rpc` client to the FE.

| RPC Method -> host cap | rpc ref | Expose under (existing control) | Rationale |
| --- | --- | --- | --- |
| `goal/set|get|list` -> `goal_set/get` | `rpc.rs:190-227` | HomeComposer goal field | Same goal surface; prefer the `/intent` custom route. |
| `checkpoint/create|list|restore` -> checkpoint family | `rpc.rs:230-282` | StateTimeline | Same checkpoint surface; prefer the `/intent` custom route. |
| `session|thread/get`, `thread/list` -> `conversation_graph` | `rpc.rs:165` | Home recent-session rows (`Home.tsx:143`) + StateTimeline lineage | Show real ancestry/children instead of mock/seeded `MOCK_SESSIONS`. |
| `item/list` -> `search_transcript` | `rpc.rs:178` | Command palette (Cmd P, `ui.tsx:72`) or Explorer search | Literal/structured transcript search from a surface the user already opens (semantic search is DEFERRED_MODEL_REQUIRED). |
| `state/inspect` -> `runtime_state` | `rpc.rs:288` | SideBar model/context popover (`SideBar.tsx:53`) | The popover already exists to show model/state and falls back to constants on mock; feed it the real snapshot. |
| `thread/fork` -> `fork_session_from_event` | `rpc.rs:137` | StateTimeline "fork from here" already fires the `fork_session` intent over `/intent` | REDUNDANT with the working intent path; keep `/intent`, do not add a second route. |
| `approval/respond` -> `ApprovalHub.decide` | `rpc.rs:300` | the gate overlay (same as `approve_effect` above) | Same decision; prefer the `/intent` custom route. |

---

## Class 3: Intentionally internal (backend-only by design, do NOT surface)

Listed for completeness so the gap set is not overstated. These are consumed by the live turn
kernel or by tests; no FE surface is warranted.

- `dispatch_tool` (`host.rs:892`), `run_command` host method (`host.rs:878`) - the live turn uses
  the kernel's own dispatcher.
- `fleet_run` (`host.rs:963`), `generate_and_publish` (`host.rs:1012`),
  `run_agent_to_terminal` (`host.rs:2257`) - test/other-host-method callers only.
- `rebuild_session_projection` (`host.rs:862`), `fork_session` seq variant (`host.rs:1055`) -
  replay/internal reuse (the FE uses `ui_events` catch-up instead).
- `job_evaluate_triggers` / `jobs_recover` - wake predicate + boot rebuild behind the job list.

---

## Class 4: Protocol surface DECLARED with no backend (cross-ref, not this file's gap)

The opposite direction (surface without backend) and covered in HIDE_SURFACE_WITHOUT_BACKEND_REPORT:
31 RPC Methods return typed NotImplemented and 4 (`agent/*`) are DEFERRED_MODEL_REQUIRED
(`rpc.rs:338-419`). They have no host binding, so there is nothing to expose. Since the FE never
calls `/rpc` anyway, they are double-dark (no backend, no caller). No action here beyond not
building FE against them.

---

## Campaign priority (drives the consolidation)

Ranked by (surface already exists) x (host cap already built + durable):

1. Verify plane -> StatusBar Problems counter. Turns a visible mock (0/0) into real receipts;
   both sides exist (`run_static_analysis`/`verification_receipts` vs `StatusBar.tsx:31`).
2. Side chat -> the two DEAD "New chat" buttons. Host-handled (`create_side_chat`), the buttons
   already exist with no onClick; pure registry + handler wiring.
3. Checkpoints -> StateTimeline. Host-handled + integrity-verified; also cures the mock ContextStack
   "snapshot state".
4. Memory -> ContextStack Memory stratum. Whole durable domain is built; the stratum already renders
   and its controls fire LOG-ONLY `pin_span`, repoint to `memory_*`.
5. Goals -> HomeComposer goal field. `goal_set/get/clear` built (`goal_evaluate` too); add the
   custom names to `wire.ts`.
6. Steering -> SteerBar. The only true end-to-end hole (nothing signals InterruptHub Steer); highest
   leverage per keystroke since the whole UI is built.
7. Workspace trust -> Add-folder flow. Without it, repo policy/instructions never activate.

Everything above reuses an existing control. No new buttons required.
