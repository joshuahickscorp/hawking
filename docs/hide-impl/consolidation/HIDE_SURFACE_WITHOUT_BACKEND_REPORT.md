# HIDE Surface Without Backend Report

Consolidation census, writer B. Every FE control that is a surface with no real backend:
`backend-unwired`, `mock`, or `dead` by census depth, PLUS the subtle class where the FE wiring is
real (`real-deep`/`real-shallow`) but the intent it fires is LOG-ONLY host-side (recorded, no host
effect). For each: the backend method that would satisfy it, or a recommendation to remove.
Grounded in the merged census catalog (branch `build/hide-impl-2026-07-19`, read-only).
UNKNOWN where the catalog is silent.

House note: hyphens and parentheses only, no long dashes.

The mirror of this file is HIDE_BACKEND_WITHOUT_SURFACE_REPORT (backends with no FE). Where a real
host cap exists, the two files agree on the same wiring.

---

## 1. Backend-unwired (control fires or stages, but the effect is plan-2 or dropped)

| Control | file:line | The gap | Backend that satisfies it / disposition |
| --- | --- | --- | --- |
| HomeComposer "Attach files" + hidden input | `home/HomeComposer.tsx:249` (input `:264`) | stages `File[]`, chips render, but `submit()` (`:153`) omits attachments, so files never reach the backend or blob store | WIRE: `submit_turn` already carries `attachments: Vec<BlobRef>` (`api.rs:9-13`). Pass the staged files as BlobRefs through a blob-upload path (via `/v1/hide/connector`). Smallest real win, the intent field already exists. UNKNOWN whether the host currently consumes BlobRefs end to end. |
| StateTimeline scrub step dots | `shell/StateTimeline.tsx:44` | `scrub_to_event` is LOG-ONLY host-side (`commands.rs:138`); component header says backend snapshots state in "plan 2" | Time-travel is FE/replay-driven by design (`replay.rs rebuild_at`). Either accept FE-side replay (then reclassify from backend-unwired) or add a host scrub effect. Likely keep FE replay, this is not a true backend gap. |
| StateTimeline "fork from here" | `shell/StateTimeline.tsx:53` | FE audit marked backend-unwired citing the same plan-2 header | RECLASSIFY: `fork_session` IS host-handled (`spawn_fork_session`, `host.rs:406`); the fork works. Only the scrub state-snapshot is plan-2. Verify fork end to end and drop the backend-unwired label. |
| Chat / HomeComposer voice mic | `Chat.tsx:172`, `home/HomeComposer.tsx:274` | records via MediaRecorder then discards; notice only | REMOVE: no transcription capability exists in the catalog (UNKNOWN if planned). |

---

## 2. Mock (renders real, nothing persisted)

| Control | file:line | Backend that satisfies it / disposition |
| --- | --- | --- |
| StatusBar Problems counter (0/0) | `shell/StatusBar.tsx:31` | WIRE to `run_static_analysis` (`host.rs:1373`) + `verification_receipts` (`host.rs:1434`). Both built, both FE-dark today. |
| ContextStack snapshot state | `surfaces/ContextStack.tsx:62` | WIRE to `checkpoint_create` (`host.rs:1629`, host-handled, integrity-verified). |
| ContextStack save skill | `surfaces/ContextStack.tsx:79` | REMOVE, or model a skill as durable `memory_add` (`host.rs:1870`). No skill store exists. |
| ContextStack load skill (x3) | `surfaces/ContextStack.tsx:115` | REMOVE the hardcoded `SKILLS` const; no backend. |

---

## 3. Dead (no handler, or never renders)

| Control | file:line | Backend that satisfies it / disposition |
| --- | --- | --- |
| StatusBar Branch item | `shell/StatusBar.tsx:27` | Bind label to `home.workspace.branch`; for switching, reserved name `switch_branch` (`wire.ts:111`) needs a host branch-switch cap (none today). Otherwise REMOVE the button chrome. |
| Chat composer Attach | `surfaces/Chat.tsx:154` | Same fix as Attach-files: `submit_turn.attachments`. Or REMOVE (duplicate of the HomeComposer attach flow). |
| DiffChipRow Accept / Reject | `chat/structure.tsx:257` / `:260` | REMOVE: never render (Conversation passes no onAccept/onReject). Real diff review is Editor DiffReview + HunkReview. |
| ChatPane / FloatingChat "New chat" | `ChatPane.tsx:79`, `FloatingChat.tsx:69` | WIRE to `create_side_chat` (`host.rs:413`, host-handled) or `new_session` (`host.rs:451`, host-handled + in registry). Do not remove, these are the ready surface for a side chat. |

---

## 4. Surface real at the FE, LOG-ONLY at the host (the subtle "no real backend" class)

These controls have real FE wiring (census depth `real-deep`/`real-shallow`) and fire a real
intent, but the host records the intent as `custom.<name>` (or `user.intent.<name>`) and takes NO
action. The surface exists; the backend effect does not.

| Control | file:line -> intent | Host status | Backend that would satisfy it |
| --- | --- | --- | --- |
| SteerBar Redirect / Steer | `SteerBar.tsx:56`/`:70` -> `redirect_run` | LOG-ONLY (`custom_names`) | Signal InterruptHub Steer; `turn/steer` is unbuilt end to end (`rpc.rs:374` NotImplemented). This is the top true hole. |
| PlanCard Approve plan | `Conversation.tsx:40` -> `approve_plan` | LOG-ONLY | Needs a host plan store + approval path; the `plan` projection is FE-registry-only with no backend emitter. |
| PlanCard Edit step / Reorder step | `Conversation.tsx:42`/`:43` -> `edit_plan_step` / `reorder_plan` | LOG-ONLY | Same plan-store gap; no host plan mutation cap exists. |
| Editor DiffReview apply all / reject | `ide/Editor.tsx:116-117` -> `accept_diff` / `reject_diff` | typed intent, recorded but NO host apply (`commands.rs:128`/`133`) | Needs a host diff-apply path. Today the agent's `edit.*` tools do the writing; `accept_diff` only records the human decision. |
| HunkReview accept / reject hunk | `ide/HunkReview.tsx:270`/`:271` -> `accept_diff` / `reject_diff` | LOG-ONLY apply + wrong granularity (no hunk id) | Needs the reserved `edit_hunk` (`wire.ts:82`) carrying a hunk id PLUS a host apply path. |
| ContextStack pin / mute / evict / note | `ContextStack.tsx:180`/`211`/`238`/`250`/`274` -> `pin_span` / `unpin_span` | LOG-ONLY | Repoint to `memory_add` / `memory_supersede` / `memory_record_outcome` (`host.rs:1870-1937`). |
| ContextStack / HomeComposer Effort | `ContextStack.tsx:136`, `HomeComposer.tsx:169` -> `switch_profile` | LOG-ONLY | No host profile-switch cap in the catalog (UNKNOWN if planned). |
| switch_model (x3) | `SideBar.tsx:43`, `Settings.tsx:41`, `HomeComposer.tsx:165` -> `switch_model` | LOG-ONLY, empty `{}` payload | No host model-switch cap in the catalog; needs one plus a real model chooser (see the duplicate report). |
| FleetView "keep best" | `FleetView.tsx:54` -> `focus_run` | LOG-ONLY | `fleet` projection is FE-registry-only; `fleet_run` is backend-only/tests. Needs a real fleet route before this means anything. |
| HomeComposer "Create PR" | `HomeComposer.tsx:162` -> `create_pr` | LOG-ONLY, optimistic notice | No host `create_pr`; could route through the agent + `shell.run gh pr create`, or add a host cap. |
| CodeActions refactor | `CodeActions.tsx:60` -> `inline_edit` | LOG-ONLY | Agent-mediated: the agent picks it up from the event and proposes an edit. Acceptable as-is. |
| Editor save file | `ide/Editor.tsx:216` -> `save_file` | LOG-ONLY | Acceptable: real persistence is the connector `fs.write_file` (`Editor.tsx:213`); the custom intent is only an agent notification. |

Two controls in this class are optimistic-but-actually-backed and only need verification, not new
backend: HomeComposer "worktree" chip (`HomeComposer.tsx:158` -> `create_worktree`) runs a real
`git worktree add` via `spawn_worktree_add` (`host.rs:447`), and Home Diff-panel hunk accept/reject
(`Home.tsx:69`) sends real `accept_diff`/`reject_diff` (which are themselves LOG-ONLY host-side, see
above). Note the census flags that `create_worktree` runs through the non-catalog unsandboxed
`spawn_exec` bypass rather than the sandboxed `git.worktree.add` tool.

---

## 5. Projection gap (FE renders projections the backend never emits)

The FE registers 23 projection names (`wire.ts:121-156`) but the backend only ever emits TWO as
`ProjectionPatch` literals: `turn` and `context_manifest`. So any FE surface bound to `plan`,
`tool`, `diff_chip`, `diff`, `editor`, `retrieval`, `memory`, `timeline`, `build`, `test`,
`diagnostics`, `sourcecontrol`, `fleet`, `run`, `merge`, `home`, `sessions`, `status`,
`turn_ended`, `plan_waiting` has NO backend emitter. These surfaces render from mock fallbacks
(`MOCK_DIFF`, `MOCK_SESSIONS`, `MOCK_HOME` in `ipc.ts`) or from untyped custom UiEvents.

Disposition: the `projection.patch -> ProjectionPatch` translation exists (`replay.rs:438-443`) but
NO host code appends `projection.patch` events. Either the host must append these events for the
projections the FE consumes, or the FE should stop registering projections it can never receive.
This is the root reason so many surfaces above fall back to mock. Note also that `api.rs` has no
`ProjectionName`/`CustomName` type, so the backend has ZERO compile-time enforcement of either
registry; both live only in the FE.

---

## 6. Protocol layer: surface declared, no backend

From the protocol census: the Method enum is a closed 48-variant set, but 31 return typed
`NotImplemented` and 4 (`agent/*`) are `DEFERRED_MODEL_REQUIRED` (`rpc.rs:338-419`). They are
mirrored in the `hide-sdk` golden `protocol.d.ts` but have no host binding. Because the FE never
posts to `/v1/hide/rpc`, they are double-dark (no backend AND no caller).

Disposition: no FE action. The typed `NotImplemented` return is honest (it never fakes success), so
these can stay as DEFERRED stubs. Do not build FE against them until a host binding lands. Deferred
groups: `workspace/*` (5), `environment/*` (4), `session/new|list|close` (3),
`thread/new|fork_ephemeral|merge_summary` (3), `turn/*` (6), `item/get|subscribe` (2),
`state/save|load|fork|release` (4), `approval/request` (1), `artifact/*` (3), `agent/*` (4).

---

## Summary

- 4 backend-unwired: 1 real win (Attach-files -> `submit_turn.attachments`), 1 reclassify (fork
  works), 1 keep-as-FE-replay (scrub), 1 remove (voice mics).
- 4 mock: 2 wire to existing built caps (Problems -> verify; snapshot -> checkpoint), 2 remove
  (skills).
- 4 dead: 2 wire (New-chat buttons), 2 remove/rebind (Branch, Chat Attach), 1 remove (DiffChipRow
  accept/reject).
- 12 real-FE-but-LOG-ONLY: 4 have real host twins to repoint at now (`memory_*`, checkpoint,
  side chat via steering-adjacent), the rest need a host cap that does not yet exist
  (`switch_model`, `switch_profile`, plan store, diff-apply, `create_pr`, fleet route) or are
  acceptable as agent-mediated (`inline_edit`, `save_file`).
- The projection gap (21 of 23 projections never emitted) is the structural root of the mock
  fallbacks; fixing it is prerequisite to trusting most read surfaces.
