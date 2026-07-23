# HIDE Workflow Before and After

Narrative receipt for the consolidation campaign, one section per high frequency workflow. BEFORE is
taken from `HIDE_PRODUCTIVITY_DENSITY_BASELINE.md` and the census set it was grounded in
(`HIDE_UI_CONTROL_CENSUS.md`, `HIDE_DEAD_DUPLICATE_CONTROL_REPORT.md`,
`HIDE_BACKEND_WITHOUT_SURFACE_REPORT.md`). AFTER is taken from the current `app/src` and the crates
it dials, read directly, and from `git diff 4fbca8bc`.

House note: hyphens and parentheses only, no long dashes.

## Measured, not counted

An earlier revision of this document reported test counts derived by grepping source for `it(` and
`test(`, and a catalog size that predated the contract remediation. Those numbers were wrong. Every
number below is from a real run.

| Measure | Reading |
| --- | --- |
| Frontend tests now | 361 passed across 18 files (`pnpm run test`) |
| Frontend tests at campaign start | 101 passed across 7 files |
| Frontend typecheck | clean (`pnpm run typecheck`) |
| Frontend build | succeeds (`pnpm run build`) |
| Rust tests | 631 passed, 0 failed, across 15 hide crates |
| Rust build | `cargo build --workspace` clean |
| Catalog commands | 52 (`crates/hide-sdk/goldens/command_catalog.json`, byte identical to `app/src/generated/command_catalog.json`) |
| Catalog binding split | 10 Intent, 41 Custom, 1 Rpc |
| Git | HEAD is the base commit `4fbca8bc`, zero commits; `app/pnpm-lock.yaml` and `app/package.json` unchanged |

Rust per crate: hide-backend 200, hide-protocol 26, hide-sdk 18, hide-serve 8, hide-kernel 74,
hide-tools 62, hide-security 40, hide-verify 15, hide-state 26, hide-compat 39,
hide-extension-registry 33, hide-program-runtime 34, hide-browser 19, hide-acp 25, hide-core 12.

## Read this first, because it changes how the numbers below should be read

The target of this campaign was never the fewest clicks. It was the fewest UNCERTAIN or REPEATED
interactions while preserving safety. Most of the click counts below did not move at all, and that is
the expected result: the baseline itself said so. The recurring density loss was not missing buttons,
it was thin wiring on top of real backend capability.

So the honest headline is not "fewer clicks". It is:

1. The SAME gesture now produces a real, durable host effect where it used to produce a log line and
   nothing else. Steering, per hunk diff review, snapshot, memory pin and every plan action were all
   one cheap gesture BEFORE. They were also all worth zero. They are worth something now.
2. Reversal exists where it did not. Ten time travel verbs, per hunk revert and re-apply, whole diff
   revert, a code rewind that really reverts the working tree, memory supersede that keeps history
   rather than deleting.
3. Evidence exists where it did not. Provenance on every hunk and every search hit, a real
   diagnostics counter instead of a hardcoded 0/0, a real branch instead of the string "main", a
   checkpoint id folded from the host record instead of scraped out of a truncated notice.
4. FOUR workflows got LONGER or narrower on purpose. Rewind takes a second confirming click and then
   an approval gate. Adding a folder ends in a trust decision. A terminal command is refused outright
   when no OS sandbox is available. Saving a file is refused under the shipped default policy. All
   four are recorded as deliberate safety costs, not regressions.

Where a thing is still model dependent or unfinished, this document says so in the same paragraph as
the win. Nothing below claims a capability that is not on the wire today.

---

## 1. Steering a running turn

**Before.** SteerBar carried its own Redirect input and a Steer button. The gesture was one keystroke
and it fired `Intent::Custom{redirect_run}`. The host recorded the custom event and took no action.
`rpc turn/steer` returned NotImplemented, and the `InterruptHub::Steer` variant sat unused right
beside Abort, Pause and Resume, which WERE wired from `cancel_run`, `pause_run` and `resume_run`. The
baseline ranked this third worst: a real button, a real intent, zero host effect.

**After.** The composer steers. While a run is in flight Enter resolves the catalog command `steer`,
now bound `Custom("redirect_run")` after the contract reconciliation pass, and `host.rs` `steer_action`
raises a real `InterruptHub::Steer`. The gesture count is unchanged at one. What changed is that the
gesture now reaches the interrupt plane, and that the SAME command serves three other entry points:
attaching a search result to a running turn (`Mod+Enter` in the palette), requesting an alternative
for a diff hunk, and the composer menu entry.

The second text box is gone. SteerBar is now a status strip carrying the run phase and the lifecycle
verbs that have no other home. One text box, one meaning.

**Not claimed.** What the model does with a mid turn steer is model dependent and is not asserted by
any test here. The wire path is what is verified.

## 2. Side chat about selected code

**Before.** Selecting code offered explain and write-tests, which inlined 600 characters of the
selection into a `submit_turn` prompt (real, agent mediated), and refactor, which fired
`Intent::Custom{inline_edit}`, a name with no host handler. A header comment advertised a fourth
action, fork and try 3 via `fleet_run`, that never rendered. There was no way to open a side chat
about a selection, and no check that the selection still said what it said when it was taken.

**After.** A selection resolves to a stable `SourceRef`: path, line range, and a content hash that is
re-read from the live buffer before every dispatch. A selection whose text has moved or changed is
STALE and refuses, instead of citing lines that no longer say what they said. That is a new class of
prevented error, not a new button. The menu dispatches catalog commands only: `submit_turn` to ask,
explain, plan, test or verify, `create_side_chat` for a review side chat, `run_command` for line
history, and the one search engine for references and attach.

Gesture count is unchanged at two (select, then pick). No permanent editor control was added: the
menu is anchored to the selection and opens on selection or Shift+F10.

**Not claimed.** The verify entry here ASKS the agent, and the label says so. The deterministic
checker is now reachable (`run_static_analysis` binds Custom and `host.rs`
`handle_static_analysis_intent` serves it), but its one control lives on the StatusBar Problems
counter. Putting a second one here would be two controls for one capability, which is the thing this
campaign removes.

## 3. Diff review

**Before.** The lowest density surface in the audited set, and the highest frequency review gesture in
the app. One keystroke per hunk, but the intent carried NO hunk id, so accepting one hunk fired a
WHOLE diff accept and a second hunk accept re-fired the identical intent. Host side, `accept_diff` and
`reject_diff` were validated and durably logged with no apply path, so even the whole diff accept
changed nothing on disk. In chat, `DiffChipRow` rendered accept and reject buttons that never appeared
because `Conversation` passed neither handler.

**After.** `accept_diff` and `reject_diff` carry an additive optional `hunk_id` (Rust
`#[serde(default)] Option<String>`, so the whole diff meaning is preserved when it is absent). With a
hunk id the host runs `apply_hunk` or `reject_hunk` through the verifying applier; without it,
`apply_diff` or `revert_diff` over the whole thing. `reject_hunk` reverts exactly that file and
appends `verify.invalidated` for every verification receipt whose scope covers the file, which is why
a rejected hunk then offers a re-verify that names the file.

The click count per hunk is unchanged at one. The granularity became real, the duplicate re-fire went
away, and the effect landed on disk. Reversal is now symmetric: an accepted hunk offers revert, a
rejected one offers re-apply, and the whole diff offers `revert_diff` once every hunk is decided. Both
directions route to the same host verbs, so one action never wears two controls.

Evidence appeared: each hunk shows its originating plan step, agent, turn and base hash when the host
recorded them, and says so plainly when it did not. The hunk header gained a detail toggle, which is
one of the campaign's two genuinely new permanent visible elements.

**The deliberate cost.** Rejecting a hunk is marked destructive and states, before it is sent, that
reverting the file invalidates every verification receipt whose scope covers it. That friction is
intentional.

**The key scoping fix.** The bare review keys (`a` accept, `r` reject) were window level. They are now
scoped: they only fire while focus is inside the diff surface, and never inside an `INPUT` or
`TEXTAREA`, so typing an `r` in the concern field or in Monaco cannot revert a file. `Escape` no
longer rejects anything; it only closes the hunk detail.

**Not claimed.** `edit_hunk` was NOT bound to a second control, even though it is a reserved name that
the earlier reports recommended. It routes to the SAME host `apply_hunk` as `accept_diff{hunk_id}`, so
binding it would have been two controls for one action. Rewind, fork an alternative and compare
candidates all need a `checkpoint_id` that no diff UiEvent carries.

## 4. Terminal continuity

**Before.** Enter dispatched `run_command` and got an ack. There was no live stdout: output only
appeared as a side effect of the unrelated `tool_progress` echo. The command ran argv UNSANDBOXED
through `exec_command_streamed`, bypassing the sandboxed `shell.run` catalog tool the agent is held
to, behind only a `dangerous_command` gate and a cwd check. The declared `pty_input` and `pty_resize`
names were never used, so Ctrl+C reached nothing. This was the lowest safety surface in the app, not
just a low density one.

**After.** The command resolves through the one spine, and the host runs it through the sandboxed
process surface (`spawn_supervised` then `confine`), which is FAIL CLOSED: with no OS sandbox it
refuses rather than running unconfined, and the terminal reads that refusal as a blocked process
instead of claiming the command ran. Output streams incrementally, every new row written as it
arrives. The store folds `tool_progress` whether or not the panel is mounted, so a process the user
navigated away from keeps running and its buffered output replays into a freshly mounted terminal.
Ctrl+C writes 0x03 to the live process stdin through `pty_input`, and geometry goes out once per
process through `pty_resize`.

One compact state row was added inside the existing panel (env, cwd, sandbox, process, exit, owning
task). It carries no buttons, and it is one of the campaign's two genuinely new permanent visible
elements. It exists because sandbox posture and cwd cannot be inferred from the scrollback.

**The deliberate cost.** A command can now be refused outright when no OS sandbox is available. Fewer
commands run. The ones that do are confined.

**Not claimed.** `stop_process`, `attach_process`, `detach_process` and `capture_process_artifact`
exist on the host but no catalog command reaches them, so this surface grows no button that would
only log. The exit field reads "not reported" because the supervisor streams output but publishes no
terminal status event yet.

## 5. Checkpoint and rewind

**Before.** The baseline called this the best return in the set and it was right. Scrub fired
`scrub_to_event`, validated and logged with no apply path. A separate permanent "fork from here"
button fired `fork_session`, which was real. ContextStack snapshot state, save skill and three
hardcoded load skill rows were toast only mocks. Meanwhile `checkpoint_create`, `checkpoint_restore`,
`checkpoint_rewind`, `replay`, `fork`, `compare` and `inspect` were all implemented, blake3 integrity
verified, and reachable by custom names that `wire.ts` refused to send.

**After.** One history menu on the timeline row resolves ten verbs, each naming a catalog command and
carrying the payload `handle_goal_checkpoint_intent` parses: create, fork from this step, inspect,
compare, replay, fork from checkpoint, restore, and three explicitly TARGETED rewinds (conversation,
code, both). A bare "rewind" is never offered, and the host now REFUSES an omitted `target` rather
than defaulting to the widest domain, because a rewind whose scope is ambiguous is the exact
uncertainty this campaign exists to remove.

A rewind also stopped being a history-only operation. A `code` or `both` rewind reverts the working
tree: every post boundary hunk is rejected newest first through the same verifying inverse write the
diff reject path uses, so the files on disk really do go back to the boundary, and a file changed
since the edit CONFLICTS and fails the rewind closed instead of being clobbered.

The control count on that row did not go up: the standalone fork button was retired and one menu
trigger took its slot. It is a one-for-one swap, and the slot now carries ten capabilities.

The gesture count for fork went from one to two (open the menu, pick the entry). That is the one place
in this document where a real capability got one click more expensive, and it bought nine sibling
capabilities that previously had no route at all.

**The deliberate cost.** Every rewind takes an explicit second confirming click, labelled "reverts
work" then "click again". Two clicks, in place, no modal. On top of that, `checkpoint_restore` and
`checkpoint_rewind` declare `approval_policy: ask`, and the host now ENFORCES that at the intent
boundary: the intent is recorded (the log is the audit trail) and the effect is parked on a gate id,
so the ack says "held for approval" and the caller can tell held from done.

**Not claimed.** The store keeps the LAST checkpoint id, not a checkpoint LIST, so this surface cannot
enumerate or diff checkpoints. A rewind's `detail.invalidated_receipts` still falls past the 200
character truncation of the Custom info notice, so invalidated receipts cannot be listed here. The
working tree revert is not transactional: a conflict part way leaves the earlier files reverted, the
same exposure `revert_diff` already has, marked in source with its upgrade path.

## 6. Search

**Before.** The Explorer field debounced a query to `code_index.search` with the payload `{q, limit}`,
a shape the connector cannot deserialize (it takes `{query: SearchQuery}` and answers `{results}`,
not a bare array). Every keystroke therefore fell through to the local filename walk, and the panel
had NEVER ONCE shown a real index hit. This is the sharpest example of an interaction that looked
dense and was not: one keystroke, an answer on screen, and the answer came from a fallback the user
had no way to identify. Separately, `rpc item/list` mapped to a real, implemented `search_transcript`
that no frontend code path could reach.

**After.** One engine in `src/ui.tsx` serves both entry points, so a hit means the same thing in the
navigator and in the palette. Real backends only: `code_index.search` for files and symbols,
`code_index.references` for references, and the `run_search` catalog command for transcript, threads
and tool output, whose hits arrive as a `search_results` Custom UiEvent. Six scopes, each declaring
its source. The local tree walk survives only as an explicitly labelled fallback.

The separate `search_transcript` catalog row was collapsed into `run_search`. The host arm already
accepted `run_search`, `search` and `search_transcript` as aliases of one implementation, so carrying
two command ids for one capability was the duplication this campaign removes. One capability, one
command id, three legacy names the host still answers to.

The click count is unchanged at one (type). What changed is that the answer is real and that its
provenance is on screen: `path:line` for a file hit, `event_id in session_id` for a log hit. Failing
legs are surfaced per scope instead of swallowed, and the other legs still return.

Acting on a hit needs no new control: Enter opens, Mod+Enter attaches to the running turn by steering
it, Mod+Shift+Enter starts a side chat forked AT the hit's event.

**Not claimed.** Semantic search is DEFERRED_MODEL_REQUIRED: `include_semantic` is pinned false and no
scope offers it. Diff hunks and plan steps are not indexed anywhere, so those origins default to the
scopes that really cover them rather than faking a scope. Correlation of a `run_search` answer is by
echoed query text, so exactly one log leg is in flight at a time; a request id on the wire would make
it exact.

## 7. Background promotion

**Before.** Nothing. `promote_run` and `resume_run_foreground` were implemented on the host and absent
from `wire.ts` CUSTOM_NAMES, so no frontend code could send them. A long task could not outlive the
foreground run.

**After.** One click promotes the live run to a durable background job, which keeps running with no
restart. Resume in foreground reattaches it by durable job id. Pause, resume and stop deliberately
reuse `pause_run`, `resume_run` and `cancel_run`, which already address a run by id, so no job verbs
were invented and no second queue exists.

Zero permanent controls were added: the row lives in the EXISTING courtyard rail, collapsed, and
unmounts when nothing is running, so an idle courtyard is byte for byte the courtyard it was. The
status bar chip likewise exists only while work is moving.

Steer and fork were pointedly NOT added here. The composer already owns steer by run id and the
timeline owns fork, so a second control would have duplicated an existing action.

**Not claimed.** `store.ts` has no jobs slice, so job lifecycle events arrive as Custom UiEvents and
are read back with a regex over the first 200 characters of the notice JSON. That shortcut is marked
in the source with its upgrade path. Job EXECUTION remains DEFERRED_MODEL_REQUIRED; create, list and
cancel governance is what is ready now.

## 8. Composer submit

**Before.** Enter fired `submit_turn` with text only. Staged attachments were chipped in the UI and
DROPPED at submit, even though `submit_turn` has always carried an attachments field on the contract.
The voice mic recorded real audio through MediaRecorder and discarded it. The Chat Attach button had
no onClick. Send relabelled itself "Queue turn" while still firing `submit_turn`, for a queue the host
does not have.

**After.** One gesture, unchanged. Staged files are now real BlobRefs carrying the file name, a real
SHA-256 digest of the bytes, size and media type, so the contract field stops being dead. The submit
control states what Enter will do right now (Start turn, Steer run, Runtime down) and is also the
submit menu, occupying the slot the retired Attach button used to hold. That slot is a one-for-one
swap: same slot, more capability. Both voice mics were removed outright.

**Not claimed.** `store.ts` `intentFor("submit_turn")` still drops an `attachments` argument, so the
courtyard composer builds that one Intent directly (the same shape the spine would build) rather than
through `runCommand`. Threading one argument closes it. This is the one remaining place where a
surface builds its own Intent for a reason other than a missing capability.

## 9. Plan operation

**Before.** The least mature high frequency domain in the baseline, and the only one where the
recommendation was "build the backend first". Approve, edit step and reorder step each fired a real
custom intent that was LOG-ONLY host side, and the `plan` projection was frontend registry only with
NO backend emitter, so the card had no feed either. The domain was contract only end to end.

**After.** `hide-backend/src/plan_domain.rs` owns a durable, mutable `PlanRecord` persisted over the KV
store, with `store_and_publish` as the single seam both the live turn emitter and the mutation
handlers route through, publishing a real `plan` projection patch. `approve_plan`, `edit_plan_step`,
`reorder_plan`, `skip_step` and `repair_step` resolve catalog commands that `handle_plan_intent`
parses and re-publishes.

The click count per action is unchanged at one. The effect is now durable, and the card shows the
declared contract (acceptance predicate, allowed effects, related files, owner agent) beside the live
state (status, verification, blocker, write gate). Availability is honest: an approved step is not
re-approved, a terminal step is not edited or skipped, only a failed step is repaired, and a write
blocked step refuses every effectful verb even if something asks for it.

**The deliberate cost.** Skip requires a reason and sends it as the host blocker, so a skipped step
cannot be silent.

**Not claimed.** WHETHER a plan appears at all is model dependent. `PlanRecord::from_kernel` reshapes a
kernel plan, so with no model producing a plan the card correctly renders nothing. The card is bound
and tested against real host shaped records, not against a live model run.

## 10. Memory pin

**Before.** Pin, mute and evict fired `pin_span` and `unpin_span`, custom names with no host handler at
all. The entire memory domain was built, durable, outcome governed and tested with no frontend route:
`memory_add`, `memory_supersede`, `memory_record_outcome`, `memory_revalidate`, `memory_context`,
`memory_get`, `memory_list`. Beside it, a Skills stratum with one save button and three hardcoded rows
toasted success and persisted nothing.

**After.** The memory stratum writes for real. Marking a claim wrong records a negative outcome
through `memory_record_outcome`, so it self quarantines. The note field adds a durable memory through
`memory_add` with the provenance the host requires. The same note supersedes the memory marked wrong
through `memory_supersede`, writing an Active plus a Superseded record rather than deleting, so a
replaced fact keeps its history. The stratum header revalidates this session's citations against disk.

Five permanent controls left: the save skill button, three hardcoded skill rows, and the model row's
`switch_profile` button (an empty payload duplicate of the one model chooser).

**Not claimed.** `memory_context`, `memory_get` and `memory_list` are reached only through the manifest
this panel already renders, not as their own controls.

## 11. Checkpoint create

**Before.** Snapshot state claimed an RWKV state snapshot for instant resume and only called
`pushNotice`. Nothing was persisted. Fork state beside it was titled "fork this state (memcpy)" and
actually dispatched `custom fleet_run {n: 2}`, spawning two text task agents, which was LOG-ONLY
anyway. `checkpoint_create` was implemented, host handled and blake3 integrity verified the whole
time.

**After.** Snapshot resolves `checkpoint_create` and seals a real boundary. Fork resolves
`fork_session` at the newest recorded step, with the memcpy claim dropped because the host replays the
prefix under a new session id, which is not a memcpy. The same `checkpoint_create` is reachable from
three places (this panel, the composer submit menu, the timeline history menu), all resolving one
command with one meaning.

Evidence closed a loop here: the host mints the id in a `checkpoint_created` Custom UiEvent, `store.ts`
folds it into `lastCheckpointId`, and every dependent verb addresses that id. This retired the regex
`readCheckpointId` stopgap that used to scrape a truncated notice string.

**Not claimed.** The store keeps the last id, not a list, so a user cannot yet pick among several
checkpoints.

## 12. Workspace trust

**Before.** Nothing, and this was the most consequential silent gap in the census. A repo enters the
host workspace graph UNTRUSTED by design, and while untrusted its instruction and policy files stay
INERT. With no trust surface anywhere in the app, a repo's CLAUDE.md and policy had no activation path
at all, and the user had no way to learn why their instructions never applied.

**After.** The add folder flow does not end at the folder. The menu stays open on an explicit trust
decision, and `workspace_set_repo_trust` resolves with the repo id the host graph keys on. Each choice
states its security MEANING, not just a label: keep untrusted leaves instruction and policy files
inert, trust makes them active for every run in that folder. Nothing is auto trusted.

**The deliberate cost.** Adding a folder is two gestures now instead of one, because the second gesture
is a security decision that should never be taken for the user. `workspace_set_repo_trust` also
declares `approval_policy: ask`, so granting trust is recorded and then parked on a gate rather than
applied straight through. Recorded as a safety cost.

**Not claimed.** No projection carries a trust VALUE back, so the status bar deliberately shows no
repository trust indicator and the chip reflects only the decision made in this session.
`workspace_add_repo`, `workspace_repo`, `workspace_graph`, `workspace_add_environment` and
`workspace_add_edge` remain without their own controls.

## 13. Saving a file from the editor

**Before.** `Mod+S` wrote through the `fs` connector AND fired a `save_file` custom intent alongside
it. No host arm consumed `save_file`, so half the gesture was a log line. The connector write itself
was a raw `std::fs::write` behind a workspace root check only. That is the part that matters: a
frontend save bypassed the permission engine and the verifying applier the agent's own edits are held
to. The human had a wider write channel than the agent, from inside the same app.

**After.** One channel. `FsConnector::write_file` dispatches `edit.write_file` through the same
permission gated `ToolDispatcher` and verifying applier the agent uses, and passes `base_hash` through
when the caller supplies it, so a concurrently changed file CONFLICTS instead of being clobbered. A
refusal comes back as `PolicyDenied` and the editor renders "save failed", never a success. The
`save_file` custom intent was retired from the wire contract.

**The deliberate cost, and it is a real one.** Under the SHIPPED default policy
(`workspace_write_default = Ask`) the dispatcher does not allow the write, so `Mod+S` is REFUSED and
nothing is written. That is a strictly worse save experience out of the box than the raw write it
replaced. It is taken on purpose: the alternative is a human write path that outranks the effect
policy. `connectors.rs` `write_file_is_refused_when_the_permission_policy_refuses_writes` asserts both
halves, the refusal and that no file appeared.

**Not claimed.** The editor does not send `base_hash` yet, so the optimistic concurrency guard is
available and unused from this surface. There is no in-app control to grant the write permission, and
none was invented: that is a config decision, not a button.

---

## Contract level remediation, folded in

These changes are not one workflow, they hold under all of them.

- **An honest negative ack.** `host.rs` keeps `HANDLED_CUSTOM_NAMES` and any `Intent::Custom` whose
  name is not on it is still RECORDED (the log is the audit trail) but is NOT reported as accepted.
  A frontend control can no longer look like it worked because the ack came back true by default.
- **Seventeen orphan custom names retired from the wire contract.** `save_file`, `inline_edit`,
  `mention_in_chat`, `quick_fix`, `queue_turn`, `rerun_step`, `fleet_run`, `resolve_conflict`,
  `pin_span`, `unpin_span`, `switch_profile`, `switch_model`, `toggle_confidence`, `focus_run`,
  `dismiss`, `create_pr`, `switch_branch`. None was handled anywhere in `crates/`, and every surface
  that used to fire them had already been retired. `wire.ts` CUSTOM_NAMES went 33 to 41: 17 out, 25
  in, and `crates/hide-backend` asserts every remaining entry has an arm.
- **`approval_policy` is enforced at the intent boundary.** `BackendHost::requires_approval` reads the
  catalog itself, so an `ask` command's effect is parked on a gate id and the ack says so. The policy
  is no longer a field the frontend displays and the backend ignores.
- **`create_worktree` is parked at the security gate.** It declares `ask` with effects
  `vcs, process, write_fs`, so the intent records and the worktree spawn waits for approval.
- **Mock data is gated to the mock transport.** The Explorer file tree and the editor's file bodies
  fell back to `MOCK_TREE` and `MOCK_FILE_BODY`. Both now go through `mockOnly(...)`, so against a
  live host they render empty rather than fabricated. A live host can no longer show, or let you
  save over, a file that does not exist.
- **`search_transcript` collapsed into `run_search`** (see section 6), and eight host handled names
  gained CommandSpecs, which is why the catalog is 52 rows and not 45.

## What did NOT get better, stated plainly

- **`goal_get` has no route.** It is the last Rpc bound catalog command, this app speaks
  `/v1/hide/intent` only, and `runCommand` throws a stated reason rather than pretending. That
  refusal is correct behavior and an incomplete capability at the same time. 51 of the 52 catalog
  commands are referenced by frontend source; this is the one that is not.
- **No live model receipt.** The `plan`, `diff` and `diagnostics` projections are emitted, and every
  claim above is verified against host shaped payloads and the wire. None of it is verified against a
  served model. Whether a plan appears, whether a steer changes the answer, and whether a refactor is
  good are all model dependent and are not asserted anywhere in this campaign.
- **Model switching does not exist.** Three empty payload buttons were retired and replaced by ONE
  component that states, in words, that this build has no model switch capability. That is honesty,
  not a feature.
- **Saving is worse out of the box.** See section 13. This is the one place the campaign made a
  common gesture fail more often, and it did so deliberately.
- **Two shortcuts are still marked in source as ponytail shortcuts with their upgrade paths**: the
  jobs regex over a truncated notice, and the single last checkpoint id in place of a list.

An earlier revision of this document claimed four custom names were still dispatched LOG-ONLY from
live controls (`save_file`, `focus_run`, `create_pr`, `switch_profile`). That is no longer true and
was corrected at the source, not in the prose: all four were retired from `wire.ts` and from the
surfaces that fired them, and `app/src/surfaces/ide/livegate.test.ts` asserts none of them is
dispatched anywhere. Zero custom names remain LOG-ONLY.
