# HIDE Productivity Density Baseline

Baseline read of the highest-frequency HIDE controls: what each can do today versus the backend capability it could reach, how many gestures a user spends to get the promised outcome, and where density (real outcome delivered per user gesture) is lowest. Grounded in the merged census catalog (branch build/hide-impl-2026-07-19). Every control below already exists on screen, so `new_visible_control_required` is false for all of them; the gaps are wiring depth, not missing buttons.

Density definition: a control is dense when one cheap gesture yields a real, complete host effect. Density is low when the gesture is cheap but the effect is log-only, optimistic, or whole-object when the user aimed at a part, or when a richer built backend capability is not routed to the control at all.

Contract facts that shape the whole read:
- Only two projections are ever emitted by the backend (`turn`, `context_manifest`). `plan`, `diff`, `tool`, `fleet`, `retrieval`, `memory`, and 15 others are FE-registry only with no backend emitter.
- 13 FE-sent custom names are LOG-ONLY (recorded as `custom.<name>` events, no host side effect): includes save_file, inline_edit, redirect_run, approve_plan, edit_plan_step, reorder_plan, fleet_run, pin_span, unpin_span, switch_profile, switch_model, focus_run, create_pr.
- accept_diff, reject_diff, scrub_to_event, open_file are typed intents that are validated and durably logged but have NO host apply path.
- checkpoint_create / checkpoint_restore / goal_set / fork_session ARE host-handled, but almost none of them are reached by a real FE control.

## Density matrix (highest-frequency controls)

### 1. Composer submit
- control: HomeComposer Send / textarea (HomeComposer.tsx:302,209); Chat composer Send / textarea (Chat.tsx:182,157)
- primary_intent: submit_turn
- current_capabilities: text-only turn; Enter or Send fires submitTurn(session_id, text). Staged file attachments are chipped in the UI but dropped at submit; the voice mic records real audio then discards it (mock).
- backend_capabilities_available: submit_turn already carries `attachments: Vec<BlobRef>` in the contract (api.rs:9-13); accepted turns run the real kernel via spawn_submit_turn_generation (host.rs:379-381).
- current_clicks_to_outcome: 1 (type + Enter)
- target: 1
- proposed_change: populate the existing `attachments` arg from the already-staged File[] (HomeComposer.tsx:249/264) so the contract field stops being dead. Text submit is dense already.
- new_visible_control_required: false

### 2. Steer / redirect a running run
- control: SteerBar Redirect input + Steer button (SteerBar.tsx:56,70)
- primary_intent: custom redirect_run
- current_capabilities: fires custom redirect_run{run_id, text}; the button and wire are real, but the host takes NO action. Mid-turn steering has no effect end to end.
- backend_capabilities_available: turn/steer is UNBUILT: rpc TurnSteer returns NotImplemented and the InterruptHub Steer variant (interrupt.rs:32) is never signaled, even though its Abort/Pause/Resume siblings ARE wired from cancel_run/pause_run/resume_run (commands.rs:89-98).
- current_clicks_to_outcome: 1 (but 0 real outcomes; log-only)
- target: 1
- proposed_change: route redirect_run to InterruptHub::Steer exactly as cancel/pause/resume already route to their variants. The signal plane exists; only the one wire is missing.
- new_visible_control_required: false

### 3. Diff accept / reject (lowest density in the audited set)
- control: Editor DiffReview apply-all / reject (Editor.tsx:117,116); HunkReview accept-hunk / reject-hunk (HunkReview.tsx:270,271); Home Diff panel hunk accept/reject (Home.tsx:69)
- primary_intent: accept_diff / reject_diff
- current_capabilities: fires accept_diff/reject_diff{run_id, diff_id}. Per-hunk accept/reject is illusory: the intent carries NO hunk id, so accepting one hunk applies the WHOLE diff and a second hunk-accept re-fires the identical intent (duplicate). Host-side the intent is validated and logged but has NO apply path, so even the whole-diff accept is inert.
- backend_capabilities_available: the contract already defines a dedicated `edit_hunk` custom name (wire.ts:82) that is never wired; the real write path is the agent's edit.* catalog tools (edit.search_replace / apply_patch / write_file), which apply and re-verify diffs but are agent-dispatch only.
- current_clicks_to_outcome: 1 keystroke per hunk (a / r), but the granularity is whole-diff and the effect is log-only
- target: 1 per hunk, with a real per-hunk apply
- proposed_change: add a hunk id to accept_diff/reject_diff (or wire the declared edit_hunk name) AND give the host an apply path (or bridge to the agent edit.* tools). Today the highest-frequency review gesture in the app delivers nothing host-side.
- new_visible_control_required: false

### 4. Code search
- control: Explorer filter / search input (Explorer.tsx:108); Explorer open search hit (Explorer.tsx:260)
- primary_intent: connector code_index.search (not an Intent)
- current_capabilities: debounced 140ms query to code_index.search{q, limit:40}; falls back to local filename match on empty/failure; on the mock transport code_index returns [] so only filename matches show. Opening a hit fires open_file(path, line).
- backend_capabilities_available: rpc item/list maps to search_transcript (literal + structured filter) and is IMPLEMENTED, but the whole /v1/hide/rpc surface is FE-dark (no app/src caller). The search.text catalog tool (ripgrep-shaped, gitignore-aware) exists but is agent-dispatch only. Semantic search is DEFERRED_MODEL_REQUIRED.
- current_clicks_to_outcome: 1 (type)
- target: 1
- proposed_change: filename/index search is dense as-is. Transcript search is a built, tested backend capability with no FE route; surfacing it would need rpc plumbing, out of the pure-wiring density scope, noted as a reach.
- new_visible_control_required: false

### 5. Editor selection actions
- control: CodeActions explain / refactor / write-tests (CodeActions.tsx:57,60,63)
- primary_intent: submit_turn (explain, write-tests); custom inline_edit (refactor)
- current_capabilities: explain and write-tests inline the selection (truncated 600 chars) into a submit_turn prompt, a real agent turn. Refactor fires custom inline_edit{instruction, selection} which is LOG-ONLY host-side. A header comment advertises a 4th action (fork & try 3 via fleet_run) that does not render (doc drift).
- backend_capabilities_available: submit_turn is real; inline_edit has no host handler, so refactor depends entirely on the agent choosing edit.* tools with no dedicated host inline-edit path. Resulting diffs are gated by the same (inert) accept_diff/reject_diff.
- current_clicks_to_outcome: 2 (select, then pick menu item)
- target: 2
- proposed_change: either give inline_edit a host handler or accept that refactor is agent-mediated like explain, and delete the fork-and-try-3 doc-drift comment. Explain/write-tests are dense; refactor is honest-but-thin.
- new_visible_control_required: false

### 6. Terminal
- control: Terminal run command on Enter (Terminal.tsx:82)
- primary_intent: run_command
- current_capabilities: dispatch and ack are real, but there is NO live stdout stream; output only appears via the unrelated tool_progress echo. The command runs argv UNSANDBOXED through exec_command_streamed (host.rs:3578), bypassing the sandboxed shell.run catalog tool, with only a dangerous_command SecurityGate and a cwd .. check. The declared pty_input / pty_resize custom names are never used.
- backend_capabilities_available: spawn_command_run streams tool_progress and parks dangerous commands behind a gate; the sandboxed shell.run catalog tool (argv-only, CATASTROPHIC deny-list, fail-closed OS sandbox, timeout watchdog) exists but the terminal does not use it.
- current_clicks_to_outcome: 1 (type + Enter)
- target: 1
- proposed_change: stream real stdout back (wire the declared pty_* names) and route the terminal through the sandboxed shell.run path so the interactive shell inherits the same safety controls the agent gets. Lowest-safety surface, not just lowest-density.
- new_visible_control_required: false

### 7. Checkpoint / rewind
- control: StateTimeline scrub step-dots / fork-from-here (StateTimeline.tsx:44,53); ContextStack snapshot-state / fork-state / save-skill / load-skill (ContextStack.tsx:62,70,79,115)
- primary_intent: scrub_to_event, fork_session, custom fleet_run (fork-state), plus toast-only mocks
- current_capabilities: scrub fires scrub_to_event (LOG-ONLY, FE/replay-driven). Fork-from-here fires fork_session which IS host-handled (spawn_fork_session + session_forked UiEvent). ContextStack snapshot-state, save-skill and load-skill are toast-only mocks that persist nothing; fork-state is mislabeled (memcpy) but actually spawns a 2-agent fleet_run text task.
- backend_capabilities_available: checkpoint_create / checkpoint_list / checkpoint_restore are IMPLEMENTED host capabilities reachable via rpc AND via the custom names checkpoint_create / checkpoint_restore (host.rs:533-556), integrity-verified with blake3. No FE control sends them today. goal_set / goal_get are likewise built and unreached.
- current_clicks_to_outcome: 1 (scrub click) / 1 (fork); snapshot and skill-load are 1 click for 0 outcome (mock)
- target: 1
- proposed_change: wire the existing ContextStack snapshot-state button to the real custom checkpoint_create and load-skill to checkpoint_restore. The durable, tested capability is sitting behind toast-only mocks; highest-value density recovery in the set with no new control.
- new_visible_control_required: false

### 8. Plan
- control: PlanCard approve / edit-step / reorder-step (structure.tsx:52,143,170)
- primary_intent: custom approve_plan / edit_plan_step / reorder_plan
- current_capabilities: all three fire real custom intents, all LOG-ONLY host-side. Approve only shows when plan.awaiting_approval.
- backend_capabilities_available: none host-handled. The `plan` projection is FE-registry only (wire.ts:124) with no backend emitter, so plan cards have no backend feed and the intents have no handler; the domain is FE-contract-only end to end in this build.
- current_clicks_to_outcome: 1 (per action)
- target: 1
- proposed_change: plan is the least-mature high-frequency domain: it needs a backend plan projection emitter AND host handlers for approve/edit/reorder before any density exists. Lowest backend maturity, ahead of wiring.
- new_visible_control_required: false

## Where density is lowest (ranked)

1. Plan (approve/edit/reorder) - no backend emitter and no host handler; FE-only end to end. Nothing works host-side.
2. Diff accept/reject - highest-frequency review gesture, but per-hunk is illusory (no hunk id, whole-diff granularity, duplicate re-fires) and host-side is log-only with no apply path.
3. Steer / redirect - real button, real intent, zero host effect; turn/steer is unbuilt though the InterruptHub Steer variant already exists next to the wired Abort/Pause/Resume.
4. Terminal - one gesture runs a command, but no live stdout and an unsandboxed exec path that skips the sandboxed shell.run tool the agent uses.
5. Checkpoint / rewind - the durable, tested checkpoint_create/restore capability is fully built and reachable via custom names, yet the FE routes those gestures into toast-only mocks. Pure wiring gap, best return.

## Where density is already healthy
- Composer text submit (submit_turn), run lifecycle controls (cancel/pause/resume via InterruptHub), fork-from-here (fork_session), gate approve/deny (approve_gate/deny_gate), new/open session (new_session/open_session), create_worktree, explain/write-tests selection actions, filename/index search, and open_file are all real one-gesture outcomes. The security gate and run-control paths are the densest surfaces in the app.

## Common cause
The recurring density loss is not missing buttons; it is thin wiring on top of real backend capability. Three patterns account for nearly all of it: (a) FE-sent custom names the host records but never acts on (13 of them), (b) built host capabilities with no FE route (checkpoint, goal, memory, transcript search), and (c) whole-object intents where the user aimed at a part (diff hunks). Every proposed change above reuses an existing control and an existing or near-existing backend path; none requires a new visible control.
