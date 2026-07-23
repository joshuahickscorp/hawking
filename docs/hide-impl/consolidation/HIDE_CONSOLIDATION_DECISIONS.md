# HIDE Consolidation Decisions

Decision record for the consolidation campaign (branch build/hide-impl-2026-07-19). One row per
retired, merged, or rewired path, with the reason and the semantic authority it defers to. Grounded
in the census set (HIDE_UI_CONTROL_CENSUS, HIDE_BACKEND_WITHOUT_SURFACE_REPORT,
HIDE_DEAD_DUPLICATE_CONTROL_REPORT, HIDE_PRODUCTIVITY_DENSITY_BASELINE). House rule: hyphens and
parentheses only, no long dashes. Living document; updated as each retirement or wiring lands.

Legend: RETIRE (remove, no honest backend or a real twin exists), MERGE (fold into a named
sibling), WIRE (point at a real host capability), KEEP (correct as-is).

---

## 1. One semantic authority per concept

The campaign requires a single owner per concept so surfaces cannot drift. Chosen owners:

| Concept | Single authority | Competing / duplicate paths to retire or defer |
| --- | --- | --- |
| Session / Thread / Turn / Item | hide-protocol semantic objects + hide-backend host records | internal ad hoc session structs stay private behind the host methods |
| Event (append-only log) | hide-backend event log + BackendReplayService | no second event representation; FE reads ui_events catch-up, not a mirror log |
| Command (every surface) | hide-protocol command_catalog (the registry, 27 commands) + hide-sdk codegen | per-surface action tables are forbidden; buttons/shortcuts/menus/palette/chat/ide resolve one CommandSpec |
| Intent (typed wire) | hide-core api.rs Intent | FE wire.ts is a generated/mirrored view, never a second source of truth |
| Method (elevated RPC) | hide-protocol Method enum + rpc.rs binding | FE never carries a hand-written method list |
| Goal | host goal_* (KV goals) | remove any FE-only goal state; HomeComposer goal field routes to goal_set/get/clear/evaluate |
| Plan | host plan projection + plan handlers (TO BUILD) | plan is FE-registry-only today; the emitter + handlers become the one authority |
| Checkpoint / Fork | host checkpoint_* (blake3 integrity) + fork_session | ContextStack snapshot/fork mocks retire in favor of these |
| Approval / Effect | hide-security policy engine + host approve/deny + effect ledger | overlay gate and inline gate share ONE handler pair |
| Tool / Capability | hide-tools catalog + hide-extension-registry ABI | terminal must route through the sandboxed shell.run tool, not a private exec path |
| Artifact | content-addressed blob/CAS | Artifacts nav has no store yet; relabel until one lands |
| Verification | hide-verify plane + host run_static_analysis / verification_receipts | StatusBar Problems reads this, not a hardcoded 0/0 |
| Memory | host memory_* (outcome-governed KV) | ContextStack pin/evict/note route here, not LOG-ONLY pin_span |
| Job | host job_* (durable, triggers) | one job list; no second queue |
| Agent | hide-kernel agents + fleet | the dormant self.kernel StubPlanner is NOT an authority (see 2.1) |

## 2. Backend code consolidations

### 2.1 Dormant StubPlanner constructor (self.kernel)

- Path: host.rs field kernel: Arc<AgentKernel> (constructed via AgentKernel::new = StubPlanner).
- Finding: OFF the live turn (verified by the prior adversarial pass). Only readers are the
  test-only fleet_run and the zero-caller run_agent_to_terminal.
- DECISION: RETIRE run_agent_to_terminal (zero callers) and remove the dormant StubPlanner
  construction; fleet_run either routes through the real KernelBuilder/RuntimePlanner path or is
  gated test-only. Preserve provenance (this record + the git history of host.rs). Implement in the
  backend code-consolidation increment; no behavior change to the live turn.

### 2.2 turn/steer NotImplemented and LOG-ONLY redirect_run

- DECISION: WIRE. redirect_run and rpc turn/steer route to the real InterruptHub Steer variant
  (backend wave 1). The LOG-ONLY handling is replaced by a real signal plus a durable steer event.

### 2.3 Custom names host-handled but unreachable

- DECISION: WIRE via handle_intent custom-dispatch arms (backend wave 1) for memory_add/supersede/
  record_outcome/revalidate, goal_evaluate, workspace_set_repo_trust, environment_switch, so the
  built host methods are reachable over /v1/hide/intent with no FE /rpc client.

## 3. Frontend control dispositions (from the dead/duplicate report)

### 3.1 Dead

| Control | file:line | Decision |
| --- | --- | --- |
| StatusBar Branch item | shell/StatusBar.tsx:27 | RETIRE the dead button chrome; render a plain label bound to home.workspace.branch (no switch_branch capability exists). |
| Chat composer Attach button | surfaces/Chat.tsx:154 | RETIRE (Chat submit drops attachments); the real attach flow is HomeComposer file input. |
| DiffChipRow Accept / Reject | surfaces/chat/structure.tsx:257,260 | RETIRE the in-chat accept/reject (never render); real review is Editor DiffReview + Home HunkReview. Keep the open/review chip. |
| ChatPane New chat | shell/ChatPane.tsx:79 | WIRE to create_side_chat (host-handled). |
| FloatingChat New chat | shell/FloatingChat.tsx:69 | WIRE to create_side_chat and MERGE both New-chat buttons onto one shared handler. |

### 3.2 Mock

| Control | file:line | Decision |
| --- | --- | --- |
| StatusBar Problems counter | shell/StatusBar.tsx:31 | WIRE to the verify plane (run_static_analysis + verification_receipts). |
| Chat Voice mic | surfaces/Chat.tsx:172 | RETIRE (no transcription capability); MERGE with the identical HomeComposer mic. |
| HomeComposer Voice mic | surfaces/home/HomeComposer.tsx:274 | RETIRE unless a real transcription backend lands (none in catalog). |
| ContextStack snapshot state | surfaces/ContextStack.tsx:62 | WIRE to checkpoint_create (integrity-verified). |
| ContextStack save skill | surfaces/ContextStack.tsx:79 | RETIRE (no skill store); a skill could later be a durable memory_add. |
| ContextStack load skill (x3) | surfaces/ContextStack.tsx:115 | RETIRE the hardcoded SKILLS list; no backend. |

### 3.3 Misleading

| Control | file:line | Decision |
| --- | --- | --- |
| Chat Send / "Queue turn" relabel | surfaces/Chat.tsx:182,186 | RETIRE the relabel; keep an honest Send. queue_turn is dead-reserved with no host queue. |
| Home rail "Artifacts" nav | surfaces/home/Home.tsx:130 | RETIRE the misleading label (it opens Code); WIRE to artifact/* only when a store lands. |
| HunkReview accept / reject hunk | ide/HunkReview.tsx:270,271 | WIRE the reserved edit_hunk (carry a hunk id) AND add a host diff-apply path; until both exist the per-hunk UI overstates granularity. |
| ContextStack fork state | surfaces/ContextStack.tsx:70 | WIRE to fork_session (host-handled); drop the fleet_run misuse and the memcpy label. |

### 3.4 Duplicate

| Cluster | Decision |
| --- | --- |
| DiffChipRow review fallback | MERGE to one open/review control; retire the paired dead accept/reject. |
| switch_model (SideBar, HomeComposer, Settings) | MERGE onto ONE model-chooser component; WIRE a host model-switch only if/when one exists, else the chooser is inert and honestly labeled. Three empty-payload copies retire. |
| approve_gate / deny_gate (overlay + inline) | KEEP two presentations; MERGE the two handler pairs into one shared approveGate/denyGate so behavior cannot drift. |

## 4. LOG-ONLY custom names (13) disposition

save_file, inline_edit, redirect_run, approve_plan, edit_plan_step, reorder_plan, fleet_run,
pin_span, unpin_span, switch_profile, switch_model, focus_run, create_pr.

- WIRE: redirect_run (to steer, wave 1); pin_span/unpin_span (to memory_add/record_outcome);
  approve_plan/edit_plan_step/reorder_plan (to the plan handlers, once the plan emitter lands);
  fleet_run misuse in fork-state repointed to fork_session.
- EVALUATE then WIRE or RETIRE: save_file (route to the edit/write tool apply path), inline_edit
  (either a host inline-edit handler or accept it is agent-mediated like explain), create_pr (a
  git tool effect), focus_run / switch_profile / switch_model (UI-state or merged chooser).
- Rule: no custom name stays LOG-ONLY after this campaign. Each is WIRED to a real effect or RETIRED
  from wire.ts with a note here.

## 5. Reserved-but-unused contract names

queue_turn, edit_hunk, revert_diff, switch_branch, pty_input, pty_resize (and any others in wire.ts).

- WIRE: edit_hunk + revert_diff (with a host diff-apply/revert path, diff increment); pty_input +
  pty_resize (with the terminal stream + sandboxed shell.run route, terminal increment).
- RETIRE: queue_turn (no host queue), switch_branch (no branch-switch capability); remove from
  wire.ts and render the affected controls honestly.

## 6. Admission gate for every wiring

Per campaign section 25, a UI wiring is admitted only when it exposes a real backend capability,
reduces effort or uncertainty, adds no unnecessary permanent control, has keyboard + palette parity
(via the command registry), shows clear state, supports reversal where relevant, preserves the
visual language, passes a deterministic workflow trace, does not duplicate another action, and reads
without documentation. Rows above that lack a real backend (plan handlers, diff-apply, model-switch,
transcript search route) are gated on the backend landing first; they are recorded here but not
wired until their capability is real.

## 7. Retirement ledger (landed)

Stage 5 (backend code consolidation) landings. One row per retired path, with the reason, the
caller confirmation, and the test outcome. Net effect: code reduced, live behavior unchanged.

| Retired path | host.rs site | Reason | Caller check | Tests |
| --- | --- | --- | --- | --- |
| `run_agent_to_terminal` (pub async fn) | removed | Zero callers anywhere in the repo (grep confirmed only its own definition); a facade over the dormant stub kernel that no live turn, server, or test reached. | grep -rn across all `*.rs`: 0 callers. | hide-backend suite green (193 passed, 0 failed). |
| `kernel: Arc<AgentKernel>` field + its `AgentKernel::new` construction in `from_services` (the dormant StubPlanner, decision 2.1) | removed | Off the live turn: the live `SubmitTurn` builds a real kernel via `build_turn_kernel` (RuntimePlanner + standard oracles). The host-held stub was an authority-looking field used only by the dead `run_agent_to_terminal` and by `fleet_run`. | Only readers were the two rows in this table. `build_turn_kernel` unaffected. | Same suite green. |
| `fleet_run` dependence on `self.kernel` | repointed | Decision 2.1 offers "route through real path or gate test-only". Fleet scheduling is model-free (drives to a terminal phase with no serve), and the sole caller is the `host_fleet_run_schedules_and_completes` test, so the launcher kernel is now built on-demand inside `fleet_run` rather than held as a dormant host field. No behavior change to the fleet path. | Sole caller: one backend test (no production or handle_intent caller; `fleet_run` is LOG-ONLY per section 4). | `host_fleet_run_schedules_and_completes` green. |

LOG-ONLY sweep (decision sections 2.2 to 4): the plan-action, diff-review, and transcript-search
custom-name arms in `handle_intent` were already wired to real effects by the earlier waves (their
"stop being log-only" comments describe completed wiring, and they route to durable host methods),
so there was no dead facade branch left to retire there.

Net line change from this stage: about -15 lines in `crates/hide-backend/src/host.rs` (a 16-line dead
function plus two field lines removed, three lines added in `fleet_run`). Gate: `cargo test -p
hide-backend --no-default-features` = 0 failed; `cargo build --workspace --no-default-features` =
clean.

## 8. Contract reconciliation (landed): Rpc-bound but custom-handled

The frontend binding pass found real drift between the catalog and the host, and it is now
reconciled. Two shapes of drift:

1. Eight commands declared `BackendBinding::Rpc` while `crates/hide-backend/src/host.rs`
   `handle_intent` already dispatched an equivalent `Intent::Custom` name for them. Because the app
   posts `/v1/hide/intent` only (a deliberate decision: no second `/rpc` client), `runCommand` threw
   on all eight, so the capability was invisible in the palette and stages reached it by building a
   raw custom intent themselves, bypassing the ONE spine. Re-bound to `Custom`: `steer` (to
   `redirect_run`), `memory_add`, `memory_supersede`, `memory_record_outcome`, `memory_revalidate`,
   `goal_evaluate`, `workspace_set_repo_trust`, `environment_switch`. The seven memory / goal-eval /
   trust / environment names were added to `wire.ts` `CUSTOM_NAMES` and its `WIRE_CUSTOM_NAMES`
   mirror. `steer` also traded its stale `steer.redirect` toolbar binding (the retired SteerBar) for
   `composer.steer`, the control that actually owns the gesture.
   Left `Rpc` on purpose, honestly unreachable: `run_static_analysis` (no custom arm in host.rs),
   `goal_get` and `search_transcript` (real `Method` strings; the reachable transcript search already
   has its own `run_search` command, so re-binding would have added a second row for one gesture).
   CORRECTION (remediation): "honestly unreachable" was not honest for `goal_get`, which kept
   `command_palette: true` while no surface could dispatch it. It is retired from the catalog, on
   the same rule that collapsed `search_transcript`. `goal/get` remains a real elevated Method.

2. Three host-handled names had NO `CommandSpec`, so surfaces dispatched them raw with no palette or
   shortcut parity: `new_session`, `revert_diff`, `edit_hunk` now have specs. `redirect_run` gets no
   spec of its own: it is the target of `steer`, and one gesture keeps one command.

Frontend consequences, kept small: `runCommand` auto-fills `session_id` for `Custom` bindings (it
already did for `Intent` bindings) unless the caller supplied one; `store.apply` keeps the last
host-minted ids off the `Custom` UiEvents it used to fold into a truncated notice
(`lastCheckpointId`, `lastForkedSession`, `lastSideChat`) instead of discarding them. That slice
retires the regex `readCheckpointId` stopgap in `StateTimeline.tsx` and unblocks the New-chat menu's
"Fork from the last checkpoint" entry, which is disabled with a stated reason until a checkpoint
exists. The steer bypasses in `Chat`, `HunkReview`, `CodeActions` and the search hit actions now go
through `runCommand("steer", ...)`, and the whole-diff revert through `runCommand("revert_diff", ...)`.

## 9. Contract cleanup (landed): the catalog, the wire list, and the last spine bypasses

The adversarial pass found the contract itself drifting. Five landings, all subtractive except the
eight specs.

### 9.1 The mirror is a mirror again

`WIRE_CUSTOM_NAMES` in `crates/hide-protocol/src/command.rs` was 17 names behind `app/src/wire.ts`,
which made the drift guard vacuous: it compared two Rust consts and passed while the real contract
moved. The guard now READS `app/src/wire.ts` and compares the parsed list in order
(`wire_custom_names_mirror_wire_ts`), so either file changing alone fails the test.

### 9.2 Seventeen orphan names RETIRED from the wire contract

A custom name lives on the contract only if `host.rs handle_intent` acts on it. These had no arm
anywhere in `crates/` and were unconditionally acked, so every control firing one was a no-op:

`save_file`, `inline_edit`, `mention_in_chat`, `quick_fix`, `queue_turn`, `rerun_step`, `fleet_run`,
`resolve_conflict`, `pin_span`, `unpin_span`, `switch_profile`, `switch_model`, `toggle_confidence`,
`focus_run`, `dismiss`, `create_pr`, `switch_branch`.

`queue_turn` and `switch_branch` were the two section 5 already ordered removed. The controls that
fired the rest were retired with them (Create PR chip, reasoning-effort cycle, fleet keep-best, the
Editor save intent, the ContextStack pin/unpin and profile/model switches). `wire.ts` carries the
rule and the retirement list in its own header.

### 9.3 search_transcript COLLAPSED into run_search

Two catalog ids for one capability. `search_transcript` was bound `Rpc(item/list)` with a
`Mod+Shift+F` that nothing could ever register (an Rpc binding is undispatchable from this app, and
a bare chord carries no query), while `run_search` was bound `Custom` for the same thing. The host
answers `run_search`, `search` and `search_transcript` on the SAME arm, so one command is the honest
count. Search opens with the palette chord and the query comes from the box.

### 9.4 Host-handled names given a CommandSpec

Each had a real `host.rs` arm and a real gesture, and each gesture built its own `Intent::Custom`
because the registry did not carry the command: `create_worktree`, `open_session`, `approve_gate`,
`deny_gate`, `approve_effect`, `deny_effect`. CORRECTION (remediation): this list read eight and
also named `compact_context` and `open_folder`, both of which are now RETIRED (empty host arm, no
reader for the record they wrote), so neither has a catalog row, a wire name or a host arm. A test
(`every_live_custom_name_has_a_command`) makes the omission impossible to repeat: every live wire
name must now have a spec.

Judged INTERNAL and deliberately left without a command: the host readers with test-only callers and
no user gesture, `verification_receipts`, `review_role_profiles` and `diff_review_receipts` (receipt
and role-profile reads that feed other host code, not a control), plus `memory_list`, `job_list`,
`job_cancel`, `stop_process`, `attach_process` and `capture_process_artifact`, which have no wire
route at all. CORRECTION (remediation): `workspace_add_repo` was on that list, and leaving it there
is what stranded the add-folder flow: with no way to put a repo in the graph,
`workspace_set_repo_trust` always addressed a node that did not exist and answered with nothing at
all. It has no command of its own, but the trust intent now carries the folder's `root_path` and
creates the node (untrusted) before applying the decision, so the flow is whole. Those are a WIRING question for a later increment, not a naming one:
giving them a spec today would put a row in the palette that `runCommand` must then refuse.

### 9.5 The stale "Rpc-bound" premise corrected

`StatusBar`, `CodeActions` and `HunkReview` each withheld a control citing an Rpc binding that the
reconciliation pass had already removed. Corrected in place: `workspace_set_repo_trust` is bound in
the add-folder flow (`HomeComposer`) where the decision is made, `environment_switch` is bound in
Settings (Workspace section), and `goal_evaluate` is bound once on the goal chip. What is genuinely
missing for the status bar is a READ (no projection carries a trust or environment value), which is
what the comments now say. `run_static_analysis` really is still Rpc and still says so.

### 9.6 Fabricated live state replaced with an honest unknown

Four surfaces printed a hardcoded model id as the live loaded model whenever no context manifest had
arrived, and the two spellings disagreed. `modelId()` in `shell/ModelChooser.tsx` is now the ONE
answer and returns `MODEL_ID_UNKNOWN` ("no model reported") when the manifest is absent, the way
`branchLabel` returns "no branch"; `StatusBar`, `ChatPane` and `FloatingChat` read it instead of
inventing one. The courtyard composer's hardcoded `"main"` git branch is gone the same way
(`branchLabel(home?.workspace?.branch)`), and `create_worktree` no longer sends that branch as its
payload: `spawn_worktree_add` creates a NEW `hide/<slug>` branch, so the old notice described an
operation the host never performs. It now says so.

### 9.7 The last spine bypasses closed

`store.ts` is the only production caller of `sendIntent` again. Converted to `runCommand`:
`Chat` pause/resume/cancel, `FleetView` stop, `Home` new session / open session / mock replay turn,
`HomeComposer` open folder / submit turn / create worktree, `autocompact` compact, and the store's
own gate approve/deny. The root cause of the last one was in the spine, not the callers:
`intentFor("submit_turn")` dropped the `attachments` argument, which is why the courtyard composer
kept a private Intent builder. That argument is threaded, so `submitTurnWith` is retired.
`noticeFailure(code)` is the shared refusal path for a fire-and-forget `runCommand`, so a guard that
throws becomes a visible notice instead of a silent no-op.

Gate: `cargo test -p hide-protocol -p hide-sdk -p hide-backend` green, `cargo build --workspace
--no-default-features` clean, `pnpm run typecheck` clean, `pnpm run test` 361 passed. Goldens
regenerated with `cargo run -p hide-sdk --bin hide-sdk-codegen` and
`crates/hide-sdk/goldens/command_catalog.json` re-copied to `app/src/generated/command_catalog.json`
(sha256 e90bcbf4).
