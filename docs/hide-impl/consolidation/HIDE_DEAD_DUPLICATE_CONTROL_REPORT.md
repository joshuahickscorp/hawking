# HIDE Dead / Duplicate / Misleading / Mock Control Report

Consolidation census, writer B. Every audited FE control whose census depth is `dead`,
`duplicate`, `misleading`, or `mock`, with `file:line` and the recommended consolidation
(retire, merge into which control, or wire to which backend). Grounded in the merged census
catalog (branch `build/hide-impl-2026-07-19`, read-only). Claims not present in the catalog are
marked UNKNOWN.

House note: this file uses hyphens and parentheses only, no long dashes.

Legend for recommendation:

- RETIRE = remove the control (no honest backend, or a real twin already exists).
- MERGE = fold into a named sibling control.
- WIRE = a real host capability exists (or a reserved contract name exists), point the control at it.

---

## 1. DEAD (no handler, or never renders)

| Control | file:line | Why dead | Recommendation |
| --- | --- | --- | --- |
| StatusBar Branch item ("main") | `shell/StatusBar.tsx:27` | styled `<button>` with no onClick; label hardcoded "main", ignores `home.workspace.branch` | WIRE or RETIRE. Reserved custom name `switch_branch` exists (`wire.ts:111`, dead-reserved) and `conversation_graph`/workspace branch data exist host-side. If no branch switcher is planned, retire the button chrome and render a plain bound label from `home.workspace.branch`. |
| Chat composer Attach button | `surfaces/Chat.tsx:154` | button has no onClick; only toggles disabled on `!ready` | MERGE into the real attach flow at `HomeComposer.tsx:249` (file input `:264`), or RETIRE. Chat `submit()` drops attachments anyway, so a lone Attach button here is dead chrome. |
| DiffChipRow Accept / Reject | `surfaces/chat/structure.tsx:257` (accept) `:260` (reject) | only render when `onAccept && onReject` are passed; `Conversation.tsx:73` passes neither, so they never render in Executor / Home / Floating chat | RETIRE the in-chat accept/reject path. Real diff review already lives in Editor DiffReview (`ide/Editor.tsx:116-117`) and Home HunkReview. Keep only the open/"review" chip. |
| ChatPane New chat button | `shell/ChatPane.tsx:79` | button has no onClick | WIRE to `create_side_chat` (host-handled at `host.rs:413`, but its custom name is absent from `wire.ts` CUSTOM_NAMES). Alternatively wire to `new_session` (`host.rs:451`, already host-handled and in the registry). This dead button is the exact ready-made surface for spawning a side chat. |
| FloatingChat New chat button | `shell/FloatingChat.tsx:69` | button has no onClick | Same as ChatPane New chat: WIRE to `create_side_chat` / `new_session`, and MERGE both New-chat buttons onto one shared handler. |

---

## 2. MOCK (renders real, no persistence or effect)

| Control | file:line | Why mock | Recommendation |
| --- | --- | --- | --- |
| StatusBar Problems counter (0 errors / 0 warnings) | `shell/StatusBar.tsx:31` | hardcoded 0/0 spans, never bound to store diagnostics | WIRE to the built-but-FE-dark verify plane: `run_static_analysis` (`host.rs:1373`) + `verification_receipts` (`host.rs:1434`). This counter is the ideal home for real problem counts (see backend-without-surface report). |
| Chat composer Voice mic | `surfaces/Chat.tsx:172` | MediaRecorder captures real audio then discards it; only pushes a "transcribing locally" notice, nothing sent or transcribed | RETIRE (no transcription capability exists in the catalog). MERGE with the identical HomeComposer voice mic mock. |
| HomeComposer Voice mic | `surfaces/home/HomeComposer.tsx:274` | records via MediaRecorder then discards; notice only | RETIRE, or gate behind a real transcription backend (none in catalog, UNKNOWN if planned). |
| ContextStack snapshot state | `surfaces/ContextStack.tsx:62` | claims RWKV state snapshot for instant resume; only calls `pushNotice`, nothing persisted | WIRE to `checkpoint_create` (`host.rs:1629`, host-handled, integrity-verified). Add the custom name to `wire.ts` and point this button at it. |
| ContextStack save skill | `surfaces/ContextStack.tsx:79` | claims to save state as a reusable skill; notice only | RETIRE, or model a "skill" as a durable `memory_add` (`host.rs:1870`). No dedicated skill store exists in the catalog (UNKNOWN if planned). |
| ContextStack load skill (rendered x3) | `surfaces/ContextStack.tsx:115` | `SKILLS` is a hardcoded const (refactor-mode, test-writing, this-repo style); "load" is notice only | RETIRE the hardcoded list; no backend exists. Same disposition as save skill. |

---

## 3. MISLEADING (label or title contradicts behavior)

| Control | file:line | The mismatch | Recommendation |
| --- | --- | --- | --- |
| Chat composer Send / "Queue turn" | `surfaces/Chat.tsx:182` (relabels at `:186`) | labeled "Queue turn" when a run is live but onClick fires `submit_turn`, not the registered custom `queue_turn` (`wire.ts:84`, defined but never used anywhere in the FE) | RETIRE the relabel (simplest): keep it honestly "Send", since `queue_turn` is dead-reserved with no host queue path. Only WIRE `queue_turn` if a real turn-queue capability is built (none in catalog). |
| Home rail "Artifacts" nav | `surfaces/home/Home.tsx:130` | onClick is `onPopToCode` (same as pip); label "Artifacts" opens no artifacts view | RETIRE the misleading label now (rename to match: it opens Code). WIRE to `artifact/*` only when an artifact store lands (`rpc.rs:417`, currently NotImplemented, no backend). |
| HunkReview accept hunk | `ide/HunkReview.tsx:270` | per-hunk gesture but `accept_diff` carries only `{run_id, diff_id}` with no hunk id, so accepting one hunk fires a WHOLE-diff accept; a second hunk-accept re-fires the same intent (duplicate) | WIRE the dedicated reserved name `edit_hunk` (`wire.ts:82`, declared, never used) carrying a hunk id, AND give the host a diff-apply path (`accept_diff` is LOG-ONLY host-side today, `commands.rs:128`). Until both exist, the per-hunk UI overstates granularity. |
| HunkReview reject hunk | `ide/HunkReview.tsx:271` | same whole-diff granularity problem; `reject_diff` has no per-hunk target | Same as accept hunk: WIRE `edit_hunk` + a host apply/revert path (`revert_diff` reserved at `wire.ts:81`, also unused). |
| ContextStack fork state | `surfaces/ContextStack.tsx:70` | titled "fork this state (memcpy)" but dispatches `custom('fleet_run', {task, n:2})`, spawning 2 text-task agents, not the constant-size state memcpy the title promises; `fleet_run` is LOG-ONLY host-side anyway | WIRE to the real fork path: `fork_session` (host-handled, `host.rs:2226`) or `state/fork` (`rpc.rs:402`, NotImplemented). `fork_session` exists today. Relabel and repoint; drop the `fleet_run` misuse. |

---

## 4. DUPLICATE / MULTI-ENTRY (same effect, several controls)

| Control | file:line | The duplication | Recommendation |
| --- | --- | --- | --- |
| DiffChipRow "review" fallback | `surfaces/chat/structure.tsx:265` | calls the same `open_file` as the file-open button; this fallback is the only branch that actually renders in chat (Conversation passes no accept/reject) | MERGE to one open/review control: keep this fallback, RETIRE the paired dead accept/reject buttons above it. |
| switch_model (three entry points) | `SideBar.tsx:107`, `HomeComposer.tsx:287`, `Settings.tsx:78` | all three fire an empty `{}` payload, no model chooser; `switch_model` is LOG-ONLY host-side (no host handler) | MERGE onto ONE reusable model-chooser component shared by all three call sites, carrying a real target-model payload. Then WIRE a host model-switch capability (none in catalog today, UNKNOWN if planned). Three empty-payload buttons are three copies of the same non-op. |
| approve_gate / deny_gate (two entry points each) | App overlay `App.tsx:161` / `:165`; InlineGate `Conversation.tsx:50` / `:54` | both entry points are real and host-handled (`host.rs:386-391`); the logic is duplicated (overlay gate vs inline chat gate) | Low priority. Acceptable as two presentations (blocking overlay vs inline card), but MERGE the two handlers into one shared `approveGate`/`denyGate` so behavior cannot drift. |

---

## Consolidation summary

- 5 dead controls: 2 retire outright (DiffChipRow accept/reject, one of the two Attach paths),
  2 wire to existing host caps (both New-chat buttons -> `create_side_chat`/`new_session`), 1 wire
  or retire (StatusBar Branch).
- 6 mock controls: 2 wire to real host caps that already exist (Problems counter -> verify plane;
  snapshot state -> `checkpoint_create`), 4 retire (both voice mics, save skill, load skill) since
  no honest backend exists.
- 5 misleading controls: 2 relabels (Queue turn, Artifacts), 1 repoint to a real fork
  (`fork_session`), 2 need the reserved `edit_hunk` name plus a host diff-apply path.
- 4 duplicate clusters: collapse the diff chip to one open control, unify `switch_model` behind one
  chooser, share the gate handlers.

The single largest consolidation win is the ContextStack "state" and "Skills" strata (snapshot /
fork / save-skill / 3x load-skill): 5 controls that are mock or misleading, of which snapshot and
fork have real host twins (`checkpoint_create`, `fork_session`) and the rest should retire.
