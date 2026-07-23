# HIDE Control Density Scorecard

Per high frequency control: what it could do BEFORE (from `HIDE_PRODUCTIVITY_DENSITY_BASELINE.md` and
the census set), what it can do NOW (from the current `app/src` and the crates it dials), its keyboard
path, its palette path, its reversal, its provenance and evidence, and a density verdict.

Density definition, unchanged from the baseline: a control is dense when one cheap gesture yields a
real, complete host effect. The target is the fewest UNCERTAIN or REPEATED interactions while
preserving safety, NOT the fewest clicks.

Verdict vocabulary:

- REAL: the gesture reaches a durable host effect, has a stated reversal or is non destructive, and
  shows evidence of what happened.
- REAL, GATED: as above, with a deliberate confirmation, approval gate or required field on a
  destructive action.
- REAL, PARTIAL: the gesture reaches a real effect, but a named piece of the workflow is still
  missing and the control says so.
- HONEST: the control cannot do the thing and states that plainly instead of pretending.
- RETIRED: the control no longer exists.

House note: hyphens and parentheses only, no long dashes.

---

## 1. Composer submit (HomeComposer send / textarea, Chat send / textarea)

| Field | Reading |
| --- | --- |
| Before | Enter fired `submit_turn` with text only. Staged attachments were chipped and dropped at submit. Send relabelled to "Queue turn" for a queue the host does not have. |
| Now | Enter sends `submit_turn` carrying real BlobRefs (name, SHA-256 content digest of the bytes, size, media type). The control states what Enter will do: Start turn, Steer run, or Runtime down. |
| Keyboard | Enter for the default action; `Mod+Enter` starts a separate new turn while a run is live (catalog `submit_turn`, toolbar binding `composer.send`). |
| Palette | `submit_turn` is a palette row; from a bare palette gesture it is filtered out because it requires a `text` argument, so it is never offered broken. |
| Reversal | None for a sent turn (correct). `cancel_run` (`Mod+.`) and `pause_run` address the run it starts. |
| Evidence | The mode label and hint state exactly what Enter does and why it cannot send when the runtime is down. Attachment chips now correspond to bytes actually transmitted. |
| Verdict | REAL, PARTIAL. `store.ts` `intentFor("submit_turn")` still drops an `attachments` argument, so the courtyard composer builds that one Intent directly rather than through `runCommand`. |

## 2. Steer control (was SteerBar redirect input, now the composer)

| Field | Reading |
| --- | --- |
| Before | Its own input plus a Steer button, firing `redirect_run`. Host recorded the event and did nothing. `turn/steer` was NotImplemented. |
| Now | The composer steers on Enter while a run is in flight, resolving catalog `steer` bound `Custom("redirect_run")`; `host.rs` `steer_action` raises a real `InterruptHub::Steer`. |
| Keyboard | Enter (default action while a run is live), `Mod+/` explicitly (catalog `steer`, toolbar binding moved from the retired `steer.redirect` to `composer.steer`). |
| Palette | Yes, `steer` is a palette row and dispatchable now that it binds Custom rather than Rpc. |
| Reversal | `cancel_run` and `pause_run` address the same run id. A steer is additive text, so there is nothing to undo. |
| Evidence | The composer label reads Steer run while a turn is in flight; the menu entry states why it cannot fire with no run. |
| Verdict | REAL. The duplicate second text box is RETIRED. What the model does with the steer is model dependent and not asserted. |

## 3. Diff accept / reject, per hunk (HunkReview)

| Field | Reading |
| --- | --- |
| Before | One key per hunk, but the intent carried no hunk id, so one hunk accept applied the WHOLE diff and a second hunk accept re-fired the identical intent. Host side had no apply path at all. |
| Now | `accept_diff` and `reject_diff` carry an additive optional `hunk_id`; with it the host runs `apply_hunk` or `reject_hunk` through the verifying applier, without it the same names mean the whole diff. |
| Keyboard | `j`/`k` or arrows move, `a` or `Mod+Enter` accepts, `r` or `Mod+Backspace` rejects, `m` opens the hunk detail, `d` opens the review detail. The bare keys are SCOPED to the diff surface and never fire inside an `INPUT` or `TEXTAREA`; `Escape` only closes the hunk detail and destroys nothing. |
| Palette | Deliberately NOT in the palette: `accept_diff` and `reject_diff` are shortcut-only rows with `context_menu` on, because from a bare palette gesture they carry no `diff_id`. |
| Reversal | Symmetric and real: accepted offers revert, rejected offers re-apply, whole diff offers `revert_diff` once every hunk is decided. Both directions route to the same host verbs. |
| Evidence | Originating plan step, agent, turn and base hash per hunk when the host recorded them, and a plain statement when it did not. Status in words. Plus and minus markers as well as pigment. |
| Verdict | REAL, GATED. Reject is marked destructive and states, before sending, that reverting the file invalidates every verification receipt covering it. |

## 4. Diff accept / reject, whole diff (Editor diff bar)

| Field | Reading |
| --- | --- |
| Before | Apply-all and reject fired `accept_diff` / `reject_diff`, both log-only host side. |
| Now | Accept-all sends the whole diff form of `accept_diff` (no `hunk_id`), which the host reads as `apply_diff`. The undo is ONE command, `revert_diff`, in both phases. The second button ("reject all", `reject_diff` with the `hunk_id` omitted) is gone: it was the same host revert under an ungated name, so pressing it walked around the approval gate on the button beside it. |
| Keyboard | Same review keys as the per hunk path, with the same surface scoping. |
| Palette | Same as above: shortcut and context menu, not palette. |
| Reversal | `revert_diff` declares `approval_policy: ask`, and the host attaches that policy to the EFFECT (`effect_command`), so a `reject_diff` with no `hunk_id` resolves to the same command and is held at the same gate whatever sends it. |
| Evidence | The bar reports the decided counts; the review detail carries the export receipt. |
| Verdict | REAL, GATED. The bar stays at two controls in both phases. |

## 5. In-chat DiffChipRow accept / reject

| Field | Reading |
| --- | --- |
| Before | Accept and reject buttons that only render when both handlers are passed; `Conversation` passed neither, so they never rendered. |
| Now | Gone. One open/review control per chip. |
| Keyboard | n/a |
| Palette | n/a |
| Reversal | n/a |
| Evidence | n/a |
| Verdict | RETIRED. Real diff review lives in the Editor diff bar and HunkReview. |

## 6. Per hunk detail toggle (HunkReview hunk header)

| Field | Reading |
| --- | --- |
| Before | Did not exist. A hunk's provenance was not shown anywhere. |
| Now | A disclosure toggle on each hunk header (`aria-expanded`) opens the hunk's provenance and its per hunk actions in place. |
| Keyboard | `m` toggles it for the selected hunk; `Escape` closes it. |
| Palette | n/a (a disclosure, not a command). |
| Reversal | n/a |
| Evidence | It IS the evidence surface: originating plan step, agent, turn, base hash, or a plain statement that the host did not record them. |
| Verdict | REAL, and one of the campaign's TWO genuinely new permanent visible elements. It is counted as an addition rather than argued away, because it is fixed per hunk chrome. |

## 7. Explorer search field

| Field | Reading |
| --- | --- |
| Before | Dialed `code_index.search` with `{q, limit}`, a shape the connector cannot deserialize, so every keystroke fell through to a local filename walk and the panel had never shown a real index hit. |
| Now | Calls the ONE engine in `src/ui.tsx`: `code_index.search` with the real `{query: SearchQuery}` shape for files and symbols, `code_index.references` for references. The tree walk survives as an explicitly labelled fallback. |
| Keyboard | Arrows and Home and End move with clamping; Enter opens; `Mod+Enter` attaches; `Mod+Shift+Enter` starts a side chat. Tree keys still drive the tree when there is no query. |
| Palette | Same engine, same hit rendering, so a result means the same thing in both entry points. |
| Reversal | n/a (a read). The actions a hit triggers are non destructive. |
| Evidence | Every hit renders its provenance (`path:line`, or `event_id in session_id`) and the source it came from. Failing legs surface per scope; the other legs still return. |
| Verdict | REAL, PARTIAL. Semantic search is DEFERRED_MODEL_REQUIRED and `include_semantic` is pinned false. The tree itself is gated to the mock transport, so a live host renders an empty tree rather than a fabricated one. |

## 8. Command palette (Mod+P)

| Field | Reading |
| --- | --- |
| Before | A command list only. |
| Now | The DERIVED catalog list (no second command table anywhere) plus the same search engine in the same box. Commands that match stay on top, hits follow, one Enter activates whichever row is selected. |
| Keyboard | `Mod+P` opens it. That chord is a SHELL binding (`toggle.palette` in `store.ts`), not a catalog command shortcut, and the palette only shows chords the shell really binds. Arrows navigate; Enter and its modifiers pick the action. |
| Palette | It is the palette. 50 of the 52 catalog rows declare `command_palette`, filtered at runtime to those that are dispatchable and self contained, so nothing is offered broken. |
| Reversal | n/a |
| Evidence | The status line reports the searching state, the result count, per leg errors, and the modifier hint. |
| Verdict | REAL. The catalog's own empty-query `run_search` row is retired: the input IS that command. The separate `search_transcript` row was collapsed into `run_search`, since the host arm answers to both names. |

## 9. Editor selection actions (CodeActions)

| Field | Reading |
| --- | --- |
| Before | Explain and write-tests inlined the selection into `submit_turn` (real). Refactor fired `inline_edit`, which has no host handler. A fourth action was advertised in a comment and never rendered. |
| Now | Every entry resolves a catalog command or the one search engine. A selection resolves to a `SourceRef` whose content hash is re-read from the live buffer before every dispatch, so a stale selection REFUSES. |
| Keyboard | Opened by selecting code or by `Shift+F10` / the context menu key. |
| Palette | The underlying command ids (`submit_turn`, `create_side_chat`, `run_command`) are palette rows. |
| Reversal | Not needed: every entry asks the agent. Whatever it proposes is reversible through the diff review path. |
| Evidence | Requests cite `path:startLine-endLine`. A stale ref is named as stale rather than sent. |
| Verdict | REAL. Refactor (`inline_edit`) and fork-and-try-3 (`fleet_run`) are RETIRED. The verify entry asks the agent and says so; the deterministic checker lives on the Problems counter, not here. |

## 10. Editor save (Mod+S)

| Field | Reading |
| --- | --- |
| Before | A raw `std::fs::write` through the `fs` connector behind a workspace root check only, PLUS a `save_file` custom intent no host arm consumed. The human write path was wider than the agent's own. |
| Now | `save_file` is a catalog command again, this time with a host arm. The editor dispatches it through `runCommand`; the host runs `FsConnector::write_file`, which dispatches `edit.write_file` through the same permission gated `ToolDispatcher` and verifying applier the agent uses. A write the policy refuses is HELD at the security gate carrying the policy's own reason, so the user can approve it. The connector route no longer accepts `write_file` at all: the only save path is the intent. |
| Keyboard | `Mod+S`, from the catalog row, and listed in the Settings keyboard table (`surfaceShortcuts`). |
| Palette | No palette row: the buffer being saved lives in the editor, so a palette gesture carries nothing to write (the argument rule). |
| Reversal | The diff review path and a code rewind reverse it. `base_hash` is the forward guard, and the editor now supplies it: `fs read_file` returns the blake3 of what it read and the save sends it back, so a concurrently changed file conflicts instead of being clobbered. |
| Evidence | "saved path", or the host's own message (the policy reason, or the applier's conflict). A `PolicyDenied` refusal is never rendered as a success and never as a generic failure. |
| Verdict | REAL and GATED, with a way through the gate. Under the shipped default (`workspace_write_default = Ask`) the first save is held, not lost: `diff_review_trace_c.rs a_refused_save_is_held_with_its_reason_and_approving_runs_it` asserts the hold, the reason, and that approving performs the write; `a_save_with_a_stale_base_hash_conflicts` asserts the guard. |

## 11. Terminal prompt

| Field | Reading |
| --- | --- |
| Before | Dispatch and ack were real; no live stdout; argv ran UNSANDBOXED through a private exec path; `pty_input` and `pty_resize` were declared and unused. |
| Now | `run_command` through the one spine; the host runs it through the sandboxed process surface, fail closed (it refuses rather than running unconfined). Output streams incrementally and replays on re-attach. `Ctrl+C` writes 0x03 through `pty_input`; geometry goes out through `pty_resize`. |
| Keyboard | Enter runs; `Ctrl+C` interrupts. |
| Palette | `run_command` is a palette row, filtered out from a bare gesture because it requires `argv`. |
| Reversal | `Ctrl+C` interrupts. No undo for a command that ran, which is correct. |
| Evidence | One state row: env, cwd, sandbox state, process id and state, exit state, owning task. A sandbox refusal reads as a blocked process, never as a success. |
| Verdict | REAL, PARTIAL, GATED. `run_command` is the one catalog row with `approval_policy: require_sandbox`. Exit state reads "not reported" because no terminal status event exists yet. `stop_process` and the attach / capture family have no catalog command, so no button claims them. |

## 12. Terminal state row

| Field | Reading |
| --- | --- |
| Before | Did not exist. Sandbox posture and cwd were not observable from the panel. |
| Now | One compact line, no buttons: workspace env, cwd, sandbox state, process id and state, exit state, owning task. |
| Keyboard | n/a (a read). |
| Palette | n/a |
| Reversal | n/a |
| Evidence | It IS the evidence surface, and it reports only what the host said. "not reported" where the host reported nothing. |
| Verdict | REAL, and the second of the campaign's TWO genuinely new permanent visible elements. Counted as an addition rather than argued away, because it is fixed chrome inside the panel. |

## 13. State timeline scrub dots

| Field | Reading |
| --- | --- |
| Before | Scrub fired `scrub_to_event`, validated and durably logged with no host apply path. |
| Now | Unchanged in shape and unchanged in wire: still `scrub_to_event`, still one click, and a failure now surfaces in the row instead of vanishing. |
| Keyboard | Each dot is a real button with an accessible name naming the step. |
| Palette | `scrub_to_event` is a palette row, filtered from a bare gesture because it requires an `event_id`. |
| Reversal | The "live" control returns to the latest state. |
| Evidence | The step message renders beside the dots; `aria-current` marks the scrubbed step. |
| Verdict | REAL, PARTIAL. This is the one high frequency gesture the campaign did NOT deepen: the host still has no scrub apply path beyond the durable log. |

## 14. State timeline history menu (replaces the fork from here button)

| Field | Reading |
| --- | --- |
| Before | One permanent button, one verb (`fork_session`, which was real). Seven host checkpoint verbs had no route at all. |
| Now | One menu trigger, ten verbs: create, fork from step, inspect, compare, replay, fork from checkpoint, restore, and three explicitly targeted rewinds (conversation, code, both). A rewind with an omitted `target` is REFUSED by the host, not defaulted to the widest domain. A code or both rewind reverts the working tree through the same verifying inverse write the diff reject path uses. |
| Keyboard | Escape closes and disarms; focus returns to the trigger. Entries carry `data-command` naming the catalog id. |
| Palette | Every entry's command id is a palette row. |
| Reversal | This control IS the reversal surface: restore leaves the source untouched, replay drops nothing, fork branches. |
| Evidence | The menu header states the addressable checkpoint id or "no checkpoint yet". Blocked entries state the reason. Fired actions report working, accepted, or the host refusal text. A code rewind reports the files reverted and the receipts invalidated. |
| Verdict | REAL, GATED. Rewinds arm on the first click and send on the second, labelled "reverts work" then "click again"; `checkpoint_restore` and `checkpoint_rewind` also declare `ask`, which the host enforces by parking the effect on a gate id. Fork went from one gesture to two, which bought nine capabilities. This slot is a one-for-one SWAP, not an addition. The working tree revert is not transactional (marked in source with its upgrade path). |

## 15. ContextStack snapshot state

| Field | Reading |
| --- | --- |
| Before | Claimed an RWKV state snapshot for instant resume; only called `pushNotice`. Nothing persisted. |
| Now | Resolves `checkpoint_create` and seals a real blake3 integrity verified boundary. |
| Keyboard | Through the palette, and through the composer submit menu entry. |
| Palette | Yes. |
| Reversal | Additive, so nothing to undo. It is what makes restore, replay, fork and rewind addressable. |
| Evidence | The host mints the id in a `checkpoint_created` Custom UiEvent, which `store.ts` folds into `lastCheckpointId`; the timeline header shows it. This retired the regex `readCheckpointId` stopgap. |
| Verdict | REAL. |

## 16. ContextStack fork state

| Field | Reading |
| --- | --- |
| Before | Titled "fork this state (memcpy)" and dispatched `custom fleet_run {n: 2}`, spawning two text task agents, log-only host side. |
| Now | Resolves `fork_session` at the newest recorded step. The memcpy claim is dropped, because the host replays the prefix under a new session id. |
| Keyboard | Through the palette. |
| Palette | Yes, filtered from a bare gesture because it requires `at_event`. |
| Reversal | A fork is additive; the source session is untouched. |
| Evidence | Disabled with a stated reason when no step has been recorded yet. |
| Verdict | REAL. |

## 17. ContextStack memory controls (pin / evict / note)

| Field | Reading |
| --- | --- |
| Before | `pin_span` and `unpin_span`, custom names with no host handler at all. Both are now retired from the wire contract entirely. |
| Now | Mark wrong writes `memory_record_outcome` (self quarantine), the note field writes `memory_add`, the same note writes `memory_supersede` when a wrong target exists, and the stratum header writes `memory_revalidate`. |
| Keyboard | Through the palette (`memory_add` also declares `context_menu`). |
| Palette | Yes for all four, now that they bind Custom rather than Rpc. |
| Reversal | Supersede keeps history (Active plus Superseded) instead of deleting, which is the reversal. |
| Evidence | The panel is a receipt: why each source is in the window from the packer's own numbers, what was excluded and why, the blake3 address of each span. A claim marked wrong renders struck through before the correction is typed. |
| Verdict | REAL. The per span pin and mute toggles that routed to `pin_span` are RETIRED. `memory_context`, `memory_get` and `memory_list` are read only through the manifest, not as their own controls. |

## 18. ContextStack Skills stratum (save skill, three load skill rows)

| Field | Reading |
| --- | --- |
| Before | One save button plus three hardcoded rows, all notice only. No skill store exists anywhere in the catalog. |
| Now | Gone. |
| Verdict | RETIRED. Four permanent controls removed. A skill could later be modelled as a durable `memory_add`; nothing pretends to be one today. |

## 19. PlanCard approve / edit step / reorder step

| Field | Reading |
| --- | --- |
| Before | Three real custom intents, all log-only, on a card with no backend emitter behind it. |
| Now | `approve_plan` (with or without a `step_id`), `edit_plan_step`, `reorder_plan` (full permutation), `skip_step` (reason required) and `repair_step` all reach `handle_plan_intent`, which mutates and republishes a durable `PlanRecord`. |
| Keyboard | Through the per step context menu (shared popover styling, no new permanent control) and the palette. |
| Palette | Yes for all five. |
| Reversal | Partial and stated: repair a failed step, skip with a blocker, edit the text. There is no plan level undo and none is claimed. |
| Evidence | Declared contract (acceptance, allowed effects, related files, owner agent) beside live state (status, verification, blocker) and the write gate, explained in words rather than by colour. |
| Verdict | REAL, GATED. Skip requires a reason. Whether a plan exists at all is model dependent, and the card is tested against host shaped records, not a live model run. |

## 20. StatusBar Problems counter

| Field | Reading |
| --- | --- |
| Before | Hardcoded `0 / 0` spans, never bound to anything, not a button. |
| Now | A real button bound to the host `diagnostics` projection (errors, warnings, per file breakdown, `last_verification_id`), AND the producer: its detail popover dispatches `run_static_analysis`, which now binds Custom and is served by `host.rs handle_static_analysis_intent`. A clean analysis reads as a real verified zero; no analysis at all reads "not run" and shows dashes rather than a fabricated zero. |
| Keyboard | The counter is a real button with `aria-expanded` and `aria-controls`; Escape closes the detail. |
| Palette | `run_static_analysis` is a palette row, filtered from a bare gesture because it requires `paths`. |
| Reversal | n/a (a read plus a re-runnable check). |
| Evidence | The popover shows the per file breakdown and the sealing verification id, or states that no receipt has been sealed. The accessible name carries counts and verification state in words, never colour. |
| Verdict | REAL. This is the one place in the app that can WRITE the diagnostics projection, which is why the trigger lives here and nowhere else. The run button is disabled with a stated reason when there is nothing to analyse. |

## 21. StatusBar Branch item

| Field | Reading |
| --- | --- |
| Before | A styled `<button>` with no onClick whose label was the literal string "main", ignoring the real branch. |
| Now | A plain label bound to `home.workspace.branch` (the host digest reads `.git/HEAD`), rendering "no branch" when the host reports none. |
| Verdict | RETIRED as a control, REAL as a read. No branch switch capability exists, so no button claims one. This retirement pairs with the Problems counter becoming a button: one-for-one, the bar's button count is unchanged. |

## 22. StatusBar background work chip

| Field | Reading |
| --- | --- |
| Before | Did not exist. |
| Now | Exists only while work is moving; counts only active and waiting runs, and its accessible name spells out each objective, state and step. |
| Verdict | REAL, and zero permanent cost: an idle bar is unchanged, so this is not counted as a permanent addition. |

## 23. New chat buttons (ChatPane header, FloatingChat header)

| Field | Reading |
| --- | --- |
| Before | Two buttons, neither with an onClick. Dead chrome in two places. |
| Now | ONE shared menu behind both headers: new thread (`new_session`), side chat inheriting history (`create_side_chat` with `inherit: true`), research fork with fresh context (same command, `inherit: false`), checkpoint this point, and fork from the last checkpoint. |
| Keyboard | `Mod+Shift+N` for `create_side_chat` (toolbar binding `chat.new`). |
| Palette | Yes for `new_session`, `create_side_chat`, `checkpoint_create` and `checkpoint_fork`. |
| Reversal | New thread clears the local transcript optimistically and the host mints the real session id; the prior session is not destroyed. |
| Evidence | Fork from the last checkpoint is disabled with a stated reason until a real checkpoint id exists. |
| Verdict | REAL. Two dead buttons became one shared handler, so the two headers cannot drift. |

## 24. Model chooser (was three switch_model buttons)

| Field | Reading |
| --- | --- |
| Before | Three entry points (SideBar popover, HomeComposer instrument row, Settings), all firing `Intent::Custom{switch_model, {}}` with an empty payload against a host that logs the name and does nothing. |
| Now | ONE `ModelChooser` component rendered by the SideBar popover and Settings, with the composer showing the same id as a plain label. It states what is loaded and that this build has no model switch capability. |
| Keyboard | n/a (a read). |
| Palette | No command, because no capability. `switch_model` and `switch_profile` are both retired from the wire contract. |
| Reversal | n/a |
| Evidence | The note is the evidence: it names the reason there is nothing to switch to. |
| Verdict | HONEST. Three permanent controls retired. When a real capability lands it gets wired here once, and all three surfaces follow. |

## 25. Home rail Artifacts item

| Field | Reading |
| --- | --- |
| Before | Labelled "Artifacts" and its onClick opened the Code chamber. No artifact store exists (`artifact/*` is NotImplemented). |
| Now | Retired from the rail. |
| Verdict | RETIRED. The Artifacts side panel tab on an active conversation is a separate surface and is not claimed here. |

## 26. Voice mics (Chat composer, HomeComposer)

| Field | Reading |
| --- | --- |
| Before | Two identical mocks. MediaRecorder captured real audio and then discarded it, pushing a "transcribing locally" notice. No transcription capability exists in the catalog. |
| Now | Gone, recorder and all. |
| Verdict | RETIRED. Two permanent controls removed. |

## 27. Add folder (HomeComposer add menu)

| Field | Reading |
| --- | --- |
| Before | Sent `open_folder`. The repo entered the host graph UNTRUSTED, and with no trust surface its instruction and policy files had no activation path. |
| Now | The flow ends in an explicit trust decision inside the same menu, dispatching `workspace_set_repo_trust` with the repo id the host graph keys on. |
| Keyboard | Through the same menu; the command is also a palette row. |
| Palette | Yes, now that `workspace_set_repo_trust` binds Custom rather than Rpc. |
| Reversal | Both states are offered and either can be sent again. |
| Evidence | Each choice states its security meaning, and the chip shows the current state including "untrusted". |
| Verdict | REAL, GATED. One extra gesture, deliberately, because it is a security decision; and the command declares `ask`, so the host parks the effect on a gate. No projection carries a trust VALUE back, so the chip reflects this session's decision only. |

## 28. Create worktree (HomeComposer add menu)

| Field | Reading |
| --- | --- |
| Before | `create_worktree` was on the wire with no CommandSpec, dispatched ad hoc from the composer. |
| Now | It has a CommandSpec and goes through `runCommand`, and its notice is derived rather than hand written. |
| Keyboard | Through the add menu and the palette (it needs no argument, so a bare palette gesture is valid). |
| Palette | Yes. |
| Reversal | None in app: removing a worktree is a git operation with no catalog command, and none was invented. |
| Evidence | The notice reports what the host returned, including the held-for-approval message. |
| Verdict | REAL, GATED. It declares `ask` with effects `vcs, process, write_fs`, so the intent is recorded and the worktree spawn is PARKED at the security gate rather than running through. |

## 29. Goal field (HomeComposer chips row)

| Field | Reading |
| --- | --- |
| Before | The host goal domain (`goal_set`, `goal_clear`, `goal_evaluate`) was built and unreached. |
| Now | The add menu binds the composer text as an acceptance condition (`goal_set`), the goal rides in the EXISTING chips row, and that chip runs the deterministic acceptance check (`goal_evaluate`). Clearing sends `goal_clear`. |
| Keyboard | Through the palette and the composer submit menu ("Update goal from this message"). |
| Palette | Yes for `goal_set`, `goal_clear` and `goal_evaluate`. `goal_get` remains Rpc bound and unreachable. |
| Reversal | `goal_clear` stops the grading. |
| Evidence | The hint states what a goal is for, so the control reads without documentation. |
| Verdict | REAL, PARTIAL. `goal_get` has no route and is the only catalog command in that state. |

## 30. Background run row (Home rail)

| Field | Reading |
| --- | --- |
| Before | Did not exist. |
| Now | `promote_run` promotes the live run to a durable job with no restart; `resume_run_foreground` reattaches by job id; pause, resume and stop reuse `pause_run`, `resume_run`, `cancel_run`. |
| Keyboard | Through the palette for each command id. |
| Palette | Yes. |
| Reversal | Stop and resume in foreground both address the same run or job, so promotion is not one way. |
| Evidence | Phase, job id, lifecycle event, blocking approval, verification read and newest process line, all in words. Each phase reads differently with styles off. |
| Verdict | REAL, PARTIAL. The row unmounts when idle, so it costs no permanent chrome. `store.ts` has no jobs slice, so the job id and lifecycle event are read back with a regex over the truncated notice JSON (marked in source with its upgrade path). Job EXECUTION is DEFERRED_MODEL_REQUIRED. |

## 31. Security gate approve / deny (shell overlay and inline capsule)

| Field | Reading |
| --- | --- |
| Before | Both entry points were real and host handled, but the handler logic was duplicated between the overlay and the inline card, so the two could drift. |
| Now | ONE pair of handlers on the store (`approveGate` / `denyGate`); both presentations call them, so an approve can never mean two different things. This is also the surface that now receives the parked effects of every `ask` command. |
| Keyboard | The overlay is a focus trapped dialog. |
| Palette | No command row (the gate is raised by the host, not invoked by the user). |
| Reversal | Deny is the reversal: it drops the held command. |
| Evidence | The gate carries the gate id it was emitted with, and it is never auto dismissed. |
| Verdict | REAL. Two presentations kept, one behavior. |

---

## Summary table: permanent controls retired versus added

A permanent control is fixed chrome that occupies space whether or not it can do anything. Menus hung
off an existing control, popovers, and elements that unmount when idle are NOT counted as permanent.

| Control | Surface | Disposition |
| --- | --- | --- |
| Redirect input | SteerBar | RETIRED (duplicate of the composer) |
| Steer button | SteerBar | RETIRED (duplicate of the composer) |
| Voice mic | Chat composer | RETIRED (mock, no transcription capability) |
| Voice mic | HomeComposer | RETIRED (mock, no transcription capability) |
| Accept | in-chat DiffChipRow | RETIRED (never rendered) |
| Reject | in-chat DiffChipRow | RETIRED (never rendered) |
| Save skill | ContextStack | RETIRED (no skill store) |
| Load skill row 1 | ContextStack | RETIRED (hardcoded) |
| Load skill row 2 | ContextStack | RETIRED (hardcoded) |
| Load skill row 3 | ContextStack | RETIRED (hardcoded) |
| Pin toggle (per span) | ContextStack | RETIRED (`pin_span`, no host handler) |
| Mute toggle (per span) | ContextStack | RETIRED (`unpin_span`, no host handler) |
| switch_profile button | ContextStack model row | RETIRED (empty payload duplicate) |
| switch_model button | SideBar popover | RETIRED (no capability) |
| switch_model button | HomeComposer | RETIRED (no capability) |
| switch_model button | Settings | RETIRED (no capability) |
| Create PR chip | HomeComposer | RETIRED (`create_pr`, no host arm) |
| Reasoning effort cycle | HomeComposer | RETIRED (`switch_profile`, no host arm) |
| Artifacts rail item | Home | RETIRED (misleading label) |
| **Attach slot** | **Chat composer** | **SWAPPED: dead Attach button out, composer mode label plus submit menu in** |
| **Branch button** | **StatusBar** | **SWAPPED: dead branch button out, Problems counter became the bar's button** |
| **Fork from here** | **StateTimeline** | **SWAPPED: one button out, one history menu trigger in (ten verbs)** |
| **Hunk detail toggle** | **HunkReview** | **ADDED (the evidence surface for a hunk's provenance)** |
| **State row** | **Terminal** | **ADDED (env, cwd, sandbox, process, exit, owner; no buttons)** |

**Reviewed accounting.** An adversarial review of the campaign independently confirmed only TWO
genuinely new permanent visible elements (the Terminal state row and the per hunk detail toggle) and
THREE one-for-one swaps, against roughly 41 retirements across all stages. The table above enumerates
19 of those retirements, the ones the high frequency workflows in this document touch. Net: roughly
minus 39 permanent controls, plus 2, with 3 slots changed hands.

Earlier revisions of this document reported "18 retired, 1 added, net minus 17" from per stage self
reports. That accounting double counted a swap as an addition, missed both genuinely new elements,
and undercounted the retirements. The reviewed numbers above supersede it.

Not counted as added, with the reason:

- The StatusBar diagnostics popover hangs off the counter that already existed.
- The background run row unmounts when nothing is running; the StatusBar background chip only exists
  while work is moving.
- The CodeActions menu is anchored to a selection, and the PlanCard step menu reuses the shared
  popover styling. Neither is fixed chrome.
- `ModelChooser` renders a note, not a button.

## Capability reach, before and after

| Measure | Before | After |
| --- | --- | --- |
| Catalog commands | 45 | 52 (`search_transcript` collapsed into `run_search`; eight host handled names gained specs) |
| Catalog binding split | mixed, with Rpc rows no surface could dial | 10 Intent, 41 Custom, 1 Rpc |
| Custom names on the wire contract (`wire.ts CUSTOM_NAMES`) | 33 | 41 (17 orphans retired, 25 added) |
| Catalog commands referenced by frontend source | not applicable (no catalog in the app) | 51 of 52 |
| Catalog commands honestly refused with a stated reason | not applicable | 1 (`goal_get`, Rpc bound) |
| Catalog rows declaring `command_palette` | not applicable | 50 of 52 |
| Custom names dispatched LOG-ONLY from a live control | 4 | 0 |
| Frontend tests (`pnpm run test`, measured) | 101 passed across 7 files | 361 passed across 18 files |
| Rust tests (measured) | not baselined | 631 passed, 0 failed, across 15 hide crates |

The 25 newly reachable custom names are the 17 that were host handled but rejected by `wire.ts`
(`create_side_chat`, `merge_side_chat`, `goal_set`, `goal_clear`, `checkpoint_create`,
`checkpoint_restore`, `approve_effect`, `deny_effect`, `skip_step`, `repair_step`,
`checkpoint_rewind`, `checkpoint_replay`, `checkpoint_fork`, `checkpoint_compare`,
`checkpoint_inspect`, `promote_run`, `resume_run_foreground`), plus the 7 re-bound from Rpc to Custom
by the contract reconciliation pass (`memory_add`, `memory_supersede`, `memory_record_outcome`,
`memory_revalidate`, `goal_evaluate`, `workspace_set_repo_trust`, `environment_switch`), plus
`run_static_analysis`, re-bound so the Problems counter can produce the projection it reads.

The 17 retired orphans: `save_file`, `inline_edit`, `mention_in_chat`, `quick_fix`, `queue_turn`,
`rerun_step`, `fleet_run`, `resolve_conflict`, `pin_span`, `unpin_span`, `switch_profile`,
`switch_model`, `toggle_confidence`, `focus_run`, `dismiss`, `create_pr`, `switch_branch`. None was
handled anywhere in `crates/`. `crates/hide-backend` asserts every remaining entry has an arm in
`HANDLED_CUSTOM_NAMES`, and any unhandled custom name now gets an honest NEGATIVE ack: recorded in
the log, not reported as accepted.

## Still unfinished, stated rather than hidden

- `goal_get` is Rpc bound and this app speaks `/v1/hide/intent` only, so `runCommand` throws a stated
  reason. Correct behavior, incomplete capability.
- Everything above is verified against host shaped payloads, the wire and the crates. Nothing is
  verified against a served model. Whether a plan appears, whether a steer changes the answer, and
  whether the agent writes a good test are model dependent and are not asserted anywhere.
- Two shortcuts remain marked in source as `ponytail:` debt with their upgrade paths: the jobs regex
  over a truncated notice, and a single last checkpoint id in place of a checkpoint list.
- The editor does not send `base_hash` on save, so the optimistic concurrency guard is available and
  unused from that surface.
