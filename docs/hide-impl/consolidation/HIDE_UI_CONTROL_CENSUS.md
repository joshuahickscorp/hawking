# HIDE UI Control Census

Complete enumeration of every user-facing frontend control in the HIDE desktop app, grounded in the merged census catalog (branch build/hide-impl-2026-07-19, read-only audit of 15 shell/home/chat FE files plus the ide/explorer/hunkreview/terminal/codeactions/contextstack surfaces against the wire.ts Intent contract).

Total controls: **108** across **26** source files. Every backend-touching control routes through a single `sendIntent(Intent)` seam (ipc.ts); there is no second outbound path, so `wired_to` is the exact Intent, custom name, connector, or tauri call each control emits.

Depth classification legend:

- **real-deep** (32): Fires a real, correctly-scoped intent/connector whose outcome is delivered
- **real-shallow** (17): Fires a real intent but thin: empty payload, optimistic-only, or log-only host effect
- **appropriate-as-is** (39): Correctly local-only shell chrome; no backend needed
- **backend-unwired** (3): FE fires an intent but the host effect is plan-2 / not yet built
- **misleading** (5): Label/title promises one thing, the wire does another
- **duplicate** (1): Second entry point to an action reachable elsewhere
- **mock** (6): Toast-only or hardcoded display; nothing is sent or persisted
- **dead** (5): No handler; the control does nothing

## Summary count by depth classification

| Depth | Count | Share |
|-------|------:|------:|
| real-deep | 32 | 30% |
| real-shallow | 17 | 16% |
| appropriate-as-is | 39 | 36% |
| backend-unwired | 3 | 3% |
| misleading | 5 | 5% |
| duplicate | 1 | 1% |
| mock | 6 | 6% |
| dead | 5 | 5% |
| **total** | **108** | **100%** |

Rollup: **71** controls are fully wired or correctly local (real-deep + appropriate-as-is), **26** are thin or wire-honest-but-inert (real-shallow + backend-unwired + misleading + duplicate), and **11** are non-functional (mock + dead).

## Controls by surface

### App.tsx (4)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Global keyboard shortcuts (Cmd P/J/B/I, Esc) | `app/src/App.tsx:118` | shortcut | Toggle palette / panel / sidebar / Executor pane; Esc closes floating chat | `local-only` | appropriate-as-is | yes | na | - |
| Security Gate Approve (blocking overlay) | `app/src/App.tsx:305` | button | Approve held gated command | `custom:approve_gate` | real-deep | yes | no | handler approveGate() App.tsx:160; auto-fired when permMode==bypass (App.tsx:97-102) |
| Security Gate Deny (blocking overlay) | `app/src/App.tsx:304` | button | Deny/drop held gated command | `custom:deny_gate` | real-deep | yes | no | handler denyGate() App.tsx:164; Esc also denies (App.tsx:283) |
| Command palette commands (6 shell view commands) | `app/src/App.tsx:171` | menu-item | Run one of 6 shell view commands | `local-only` | appropriate-as-is | yes | na | - |

### ui.tsx (1)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Command palette input + result list (Cmd P) | `app/src/ui.tsx:72` | other | Filter and select a command; arrows/enter navigate | `local-only` | appropriate-as-is | yes | na | - |

### Toolbar.tsx (6)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Toggle navigator (Explorer sidebar) | `app/src/shell/Toolbar.tsx:39` | toggle | Show/hide the Explorer sidebar | `local-only` | appropriate-as-is | yes | na | - |
| Chamber tabs Chat / Code | `app/src/shell/Toolbar.tsx:52` | tab | Switch between Chat and Code chambers | `local-only` | appropriate-as-is | yes | na | - |
| Cancel run (Stop) | `app/src/shell/Toolbar.tsx:70` | button | Cancel the active run | `cancel_run` | real-deep | yes | no | onCancel -> App.cancelRun() App.tsx:153; disabled unless working; Code chamber only |
| Executor toggle (sparkle, Cmd I) | `app/src/shell/Toolbar.tsx:88` | toggle | Show/hide the Executor chat pane | `local-only` | appropriate-as-is | yes | na | - |
| Settings (gear) | `app/src/shell/Toolbar.tsx:98` | button | Open the Settings dialog | `local-only` | appropriate-as-is | yes | na | - |
| Toggle panel (Cmd J) | `app/src/shell/Toolbar.tsx:101` | toggle | Show/hide the editor bottom panel/terminal | `local-only` | appropriate-as-is | yes | na | - |

### SideBar.tsx (2)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Model workcard button (open context popover) | `app/src/shell/SideBar.tsx:53` | button | Open the model/context popover (ceiling, .tq multiplier, recall, state) | `local-only` | real-shallow | yes | yes | popover reads live manifest; on mock/no-host falls back to constants qwen2.5 / loaded, cached |
| 'switch model' (in model popover) | `app/src/shell/SideBar.tsx:107` | button | Request a model switch | `custom:switch_model` | real-shallow | yes | no | handler SideBar.tsx:42 fires empty payload {} + local notice; no model chooser, no target model |

### StatusBar.tsx (2)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Branch item ('main') | `app/src/shell/StatusBar.tsx:27` | status-element | (none) styled as button but no handler | `none/dead` | dead | yes | no | <button> with no onClick; branch label hardcoded 'main' ignoring home.workspace.branch |
| Problems counter (0 errors / 0 warnings) | `app/src/shell/StatusBar.tsx:31` | status-element | (display only) error/warning counts | `none/dead` | mock | no | no | hardcoded 0/0 spans, never bound to store diagnostics |

### StateTimeline.tsx (3)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Step dots (scrub to state) | `app/src/shell/StateTimeline.tsx:44` | other | Scrub agent state to a past tool step | `scrub_to_event` | backend-unwired | yes | no | handler scrub() line 21; file header says backend snapshots state in plan 2; keyed on tool call_id as event_id |
| 'live' (return to latest) | `app/src/shell/StateTimeline.tsx:51` | button | Deselect scrub, return to newest state | `local-only` | appropriate-as-is | yes | na | - |
| 'fork from here' | `app/src/shell/StateTimeline.tsx:53` | button | Fork a new session/branch from selected state | `fork_session` | backend-unwired | yes | no | handler forkHere() line 25; same plan-2 backend gap as scrub |

### Chat.tsx (4)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Chat composer: Attach button | `app/src/surfaces/Chat.tsx:154` | composer-action | (intended attach) no-op | `none/dead` | dead | yes | no | button has no onClick; only toggles disabled on !ready |
| Chat composer: textarea (Enter submits) | `app/src/surfaces/Chat.tsx:157` | composer-action | Type a task; Enter submits, Shift+Enter newline | `submit_turn` | real-deep | yes | no | submit() Chat.tsx:50 -> intent.submitTurn(sessionId,text), no attachments |
| Chat composer: Voice mic (record toggle) | `app/src/surfaces/Chat.tsx:172` | composer-action | Start/stop local mic recording | `local-only` | mock | yes | no | MediaRecorder captures real audio then discarded; only a transcribing locally notice, nothing sent/transcribed |
| Chat composer: Send / 'Queue turn' | `app/src/surfaces/Chat.tsx:182` | composer-action | Send (or when live 'Queue turn') | `submit_turn` | misleading | yes | no | labeled Queue turn when live but onClick=submit fires submit_turn, not the registered custom queue_turn (wire.ts:84, never used) |

### SteerBar.tsx (5)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Redirect input (Enter fires) | `app/src/surfaces/chat/SteerBar.tsx:56` | composer-action | Type steering text to redirect the run | `custom:redirect_run` | real-deep | yes | no | onRedirect wired in Chat.tsx:59 -> custom redirect_run{run_id,text} |
| Steer button | `app/src/surfaces/chat/SteerBar.tsx:70` | button | Send redirect text to running run | `custom:redirect_run` | real-deep | yes | no | - |
| Resume button | `app/src/surfaces/chat/SteerBar.tsx:74` | button | Resume a paused run | `resume_run` | real-deep | yes | no | onResume Chat.tsx:61 |
| Pause button | `app/src/surfaces/chat/SteerBar.tsx:79` | button | Pause the active run | `pause_run` | real-deep | yes | no | onPause Chat.tsx:60 |
| Cancel button | `app/src/surfaces/chat/SteerBar.tsx:82` | button | Cancel the active run | `cancel_run` | real-deep | yes | no | onCancel Chat.tsx:62 |

### Conversation.tsx (1)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Per-codeblock Copy button | `app/src/surfaces/chat/Conversation.tsx:120` | button | Copy code block to clipboard | `local-only` | appropriate-as-is | yes | na | injected into each <pre> via navigator.clipboard |

### structure.tsx (9)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| PlanCard: Approve plan | `app/src/surfaces/chat/structure.tsx:52` | button | Approve a suggest-only plan | `custom:approve_plan` | real-deep | yes | no | approvePlan Conversation.tsx:40; only shown when plan.awaiting_approval |
| PlanCard: Edit step (inline input) | `app/src/surfaces/chat/structure.tsx:143` | editor-action | Rename a plan step | `custom:edit_plan_step` | real-deep | yes | no | editStep Conversation.tsx:41; commit on Enter/blur |
| PlanCard: Reorder step up/down | `app/src/surfaces/chat/structure.tsx:170` | button | Move a plan step up or down | `custom:reorder_plan` | real-deep | yes | no | reorder Conversation.tsx:43; up 170 / down 175 |
| ToolChip row: expand/collapse | `app/src/surfaces/chat/structure.tsx:198` | toggle | Expand a tool call to show its call_id | `local-only` | appropriate-as-is | yes | na | - |
| DiffChipRow: Open file | `app/src/surfaces/chat/structure.tsx:241` | diff-action | Open the diff's file to review in the editor | `open_file` | real-deep | yes | no | openDiff Conversation.tsx:45 -> intent.openFile(path) + onOpenDiff chamber switch |
| DiffChipRow: 'review' fallback | `app/src/surfaces/chat/structure.tsx:265` | diff-action | Open file to review (fallback when no accept/reject) | `open_file` | duplicate | yes | no | calls same onOpen as file button; this branch is what actually renders since Conversation passes no onAccept/onReject |
| DiffChipRow: Accept / Reject (inline chat) | `app/src/surfaces/chat/structure.tsx:257` | diff-action | Accept/reject a diff chip inline | `accept_diff` | dead | yes | no | only rendered when onAccept&&onReject passed; Conversation.tsx:73 passes neither, so these never render in Executor/Home/Floating chat (accept 257 / reject 260) |
| InlineGate: Approve | `app/src/surfaces/chat/structure.tsx:300` | button | Approve the inline security gate | `custom:approve_gate` | real-deep | yes | no | approveGate Conversation.tsx:49; duplicate entry point to App overlay gate |
| InlineGate: Dismiss | `app/src/surfaces/chat/structure.tsx:303` | button | Deny/dismiss the inline gate | `custom:deny_gate` | real-deep | yes | no | denyGate Conversation.tsx:53 |

### ChatPane.tsx (5)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Resize handle (drag left edge) | `app/src/shell/ChatPane.tsx:65` | other | Drag to resize the docked Executor pane | `local-only` | appropriate-as-is | no | na | pointer-only, width persisted to localStorage |
| New chat button | `app/src/shell/ChatPane.tsx:79` | button | (intended new chat) no-op | `none/dead` | dead | yes | no | button has no onClick |
| Open in Chat (pip) | `app/src/shell/ChatPane.tsx:83` | button | Pop the conversation into the Chat chamber | `local-only` | appropriate-as-is | yes | na | - |
| Float button | `app/src/shell/ChatPane.tsx:87` | button | Undock Executor into a floating panel | `local-only` | appropriate-as-is | yes | na | - |
| Close button | `app/src/shell/ChatPane.tsx:92` | button | Close the Executor pane | `local-only` | appropriate-as-is | yes | na | - |

### FloatingChat.tsx (5)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Header drag (move) | `app/src/shell/FloatingChat.tsx:57` | other | Drag the floating Executor by its header | `local-only` | appropriate-as-is | no | na | - |
| New chat button | `app/src/shell/FloatingChat.tsx:69` | button | (intended new chat) no-op | `none/dead` | dead | yes | no | button has no onClick |
| Open in Chat (pip) | `app/src/shell/FloatingChat.tsx:72` | button | Pop into the Chat chamber | `local-only` | appropriate-as-is | yes | na | - |
| Dock button | `app/src/shell/FloatingChat.tsx:75` | button | Dock the floating Executor to the side | `local-only` | appropriate-as-is | yes | na | - |
| Close button | `app/src/shell/FloatingChat.tsx:78` | button | Close the floating Executor | `local-only` | appropriate-as-is | yes | na | - |

### Home.tsx (7)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Home rail: Chamber tabs Chat / Code | `app/src/surfaces/home/Home.tsx:110` | tab | Switch chamber from the home rail | `local-only` | appropriate-as-is | yes | na | - |
| Home rail: New session | `app/src/surfaces/home/Home.tsx:127` | button | Clear transcript and start a fresh session | `custom:new_session` | real-deep | yes | no | newSession() Home.tsx:82 = local startNewSession() + custom new_session intent |
| Home rail: 'Artifacts' nav | `app/src/surfaces/home/Home.tsx:130` | button | Labeled Artifacts, actually pops to Code chamber | `local-only` | misleading | yes | na | onClick=onPopToCode (same as pip); label Artifacts does not open an artifacts view |
| Home rail: 'Customize' nav | `app/src/surfaces/home/Home.tsx:133` | button | Open Settings | `local-only` | appropriate-as-is | yes | na | - |
| Home rail: Recent session rows | `app/src/surfaces/home/Home.tsx:143` | button | Open a recent session in place | `custom:open_session` | real-shallow | yes | no | openSession() Home.tsx:89; on mock replays via submit_turn, on live fires custom open_session{session_id}; sessions list is mock/seeded |
| Home stage: Panel switcher tabs (Terminal/Diff/Preview/Tools/Artifacts) | `app/src/surfaces/home/Home.tsx:170` | tab | Toggle which side panel shows beside the conversation | `local-only` | appropriate-as-is | yes | na | - |
| Diff panel hunk Accept / Reject (via HunkReview) | `app/src/surfaces/home/Home.tsx:69` | diff-action | Accept/reject a diff hunk in the Diff panel | `accept_diff` | real-shallow | partial | no | onDiffAct Home.tsx:69-74 -> intent.acceptDiff/rejectDiff; applies status optimistically then sends; buttons in ide/HunkReview (a/r hotkeys); diff falls back to MOCK_DIFF on mock |

### HomeComposer.tsx (13)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| 'worktree' chip | `app/src/surfaces/home/HomeComposer.tsx:187` | button | Create an isolated git worktree on the branch | `custom:create_worktree` | real-shallow | yes | no | createWorktree() line 157 fires intent + optimistic notice regardless of ack |
| 'Create PR' chip | `app/src/surfaces/home/HomeComposer.tsx:190` | button | Open a pull request for the branch | `custom:create_pr` | real-shallow | yes | no | createPr() line 161 fires intent + optimistic opening pull request notice, no ack check |
| Attachment chip remove (x) | `app/src/surfaces/home/HomeComposer.tsx:201` | button | Remove a staged attachment | `local-only` | appropriate-as-is | yes | na | - |
| Composer textarea (Enter submits) | `app/src/surfaces/home/HomeComposer.tsx:209` | composer-action | Describe a task; Enter submits | `submit_turn` | real-deep | yes | no | submit() line 146 -> intent.submitTurn(sessionId,text) WITHOUT attachments |
| Permission mode cycle (Ask/Auto/Bypass) | `app/src/surfaces/home/HomeComposer.tsx:228` | toggle | Cycle the gate permission mode | `local-only` | real-shallow | yes | no | onPermMode only sets client state; bypass drives App auto-approve (App.tsx:97), reset to ask on restart; never sent to host |
| Add context (+) menu | `app/src/surfaces/home/HomeComposer.tsx:232` | menu-item | Open the add-folder/attach-files menu | `local-only` | appropriate-as-is | yes | na | - |
| 'Add folder' menu item | `app/src/surfaces/home/HomeComposer.tsx:245` | menu-item | Pick a workspace folder and set it | `custom:open_folder` | real-shallow | yes | no | addFolder() line 54 -> Tauri dialog.open then custom open_folder{path}; on web/no-Tauri degrades to an info notice |
| 'Attach files' menu item + hidden file input | `app/src/surfaces/home/HomeComposer.tsx:249` | composer-action | Pick files to attach to the turn | `local-only` | backend-unwired | yes | no | file input line 264; addFiles stages File[] and chips render, but submit() (line 153) omits attachments so files never reach the backend/blob store |
| Voice mic | `app/src/surfaces/home/HomeComposer.tsx:274` | composer-action | Start/stop local mic recording | `local-only` | mock | yes | no | records via MediaRecorder then discards; only a transcribing locally notice |
| Model button (switch model) | `app/src/surfaces/home/HomeComposer.tsx:287` | button | Request a model switch | `custom:switch_model` | real-shallow | yes | no | switchModel() line 165 fires empty payload; no model chooser |
| Effort button (cycle Standard/Extra/Max) | `app/src/surfaces/home/HomeComposer.tsx:290` | toggle | Cycle reasoning effort profile | `custom:switch_profile` | real-shallow | yes | no | cycleEffort() line 166 sets local label + fires switch_profile{profile} |
| Open in Code (pip) | `app/src/surfaces/home/HomeComposer.tsx:293` | button | Open the conversation in the Code chamber | `local-only` | appropriate-as-is | yes | na | - |
| Send | `app/src/surfaces/home/HomeComposer.tsx:302` | composer-action | Submit the task turn | `submit_turn` | real-deep | yes | no | same submit(); attachments dropped |

### Digest.tsx (1)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Range tabs (All / 30d / 7d) | `app/src/surfaces/home/Digest.tsx:60` | tab | Slice the activity heatmap window | `local-only` | appropriate-as-is | yes | na | heatmap data is mock/seeded from host digest |

### Preview.tsx (3)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Reload | `app/src/surfaces/home/Preview.tsx:27` | button | Force the preview iframe to reload | `local-only` | appropriate-as-is | yes | na | - |
| URL input (Enter loads) | `app/src/surfaces/home/Preview.tsx:32` | composer-action | Type a local dev-server URL | `local-only` | appropriate-as-is | yes | na | - |
| Go | `app/src/surfaces/home/Preview.tsx:43` | button | Load the URL into the iframe | `local-only` | appropriate-as-is | yes | na | - |

### ChatPanel.tsx (1)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Close panel | `app/src/surfaces/home/ChatPanel.tsx:79` | button | Close the active side panel | `local-only` | appropriate-as-is | yes | na | - |

### Settings.tsx (3)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Close (x / backdrop / Esc) | `app/src/surfaces/Settings.tsx:69` | button | Close the Settings dialog | `local-only` | appropriate-as-is | yes | na | Esc handler line 33, backdrop click line 57 |
| 'switch model' | `app/src/surfaces/Settings.tsx:78` | button | Request a model switch | `custom:switch_model` | real-shallow | yes | yes | switchModel() line 40 empty payload + notice; section exposes model/status/endpoint provenance rows; third switch_model entry point |
| 'check for updates' | `app/src/surfaces/Settings.tsx:94` | button | Check for an app update | `tauri-ipc:updater.check` | real-shallow | yes | no | checkUpdates() line 48 -> checkForUpdate() (shell/updater.ts) via window.__TAURI__.updater; on web/dev degrades to managed by the desktop app |

### FleetView.tsx (2)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| 'keep best' | `app/src/surfaces/fleet/FleetView.tsx:69` | button | Focus/keep one fleet branch, discard others | `custom:focus_run` | real-shallow | yes | no | keep() line 54 fires focus_run{run_id}; fleet runs are optimistically seeded (App/mock), not real forks yet (plan 2) |
| 'stop' | `app/src/surfaces/fleet/FleetView.tsx:72` | button | Stop one fleet attempt | `cancel_run` | real-deep | yes | no | stop() line 55 -> intent.cancelRun(run.id); disabled unless active |

### CodeActions.tsx (4)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Explain selection | `app/src/surfaces/ide/CodeActions.tsx:57` | menu-item | Dispatch submit_turn with 'Explain this code:' + selection (truncated 600 chars), then toast + close | `submit_turn` | real-deep | yes | no | popover reveals on editor selection (Editor.tsx onDidChangeCursorSelection L219); menu keyboard focus/Up/Down/Enter |
| Refactor selection | `app/src/surfaces/ide/CodeActions.tsx:60` | menu-item | Dispatch custom inline_edit{instruction:refactor, selection} so agent proposes an inline edit | `custom:inline_edit` | real-deep | yes | no | - |
| Write tests for selection | `app/src/surfaces/ide/CodeActions.tsx:63` | menu-item | Dispatch submit_turn with 'Write tests for this code:' + selection, then toast + close | `submit_turn` | real-deep | yes | no | - |
| Dismiss (Esc) | `app/src/surfaces/ide/CodeActions.tsx:35` | shortcut | Esc closes the popover and returns focus to the editor (onDone) | `local-only` | appropriate-as-is | yes | na | header advertises a 4th action 'fork & try 3' via fleet_run but only explain/refactor/test render (doc drift) |

### Editor.tsx (4)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Save file (Cmd/Ctrl+S) | `app/src/surfaces/ide/Editor.tsx:211` | editor-action | Write buffer via connector fs.write_file, toast, then dispatch custom save_file{path} to notify the agent loop | `custom:save_file` | real-deep | yes | no | actual persistence is connector fs.write_file (L213); custom:save_file intent is only the agent notification |
| DiffReview: apply all | `app/src/surfaces/ide/Editor.tsx:117` | diff-action | acceptAll: clear pending diff optimistically, dispatch accept_diff(run_id,diff_id) for the whole diff | `accept_diff` | real-deep | yes | no | keyboard Tab (onKeyDown L83-87), suppressed while editing Monaco hidden textarea; diff fed by real projection_patch{diff} |
| DiffReview: reject | `app/src/surfaces/ide/Editor.tsx:116` | diff-action | rejectAll: clear pending diff, dispatch reject_diff(run_id,diff_id) | `reject_diff` | real-deep | yes | no | keyboard Esc (onKeyDown L89) |
| DiffPane: inline/side-by-side view toggle | `app/src/surfaces/ide/Editor.tsx:262` | toggle | Flip Monaco DiffEditor renderSideBySide (local sideBySide state L39) | `local-only` | appropriate-as-is | yes | na | legitimately client-only view state; drives the Monaco option |

### HunkReview.tsx (3)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Accept hunk | `app/src/surfaces/ide/HunkReview.tsx:270` | diff-action | Flip hunk status to accepted and (via Editor onAct L102-106) dispatch accept_diff(run_id,diff_id) | `accept_diff` | misleading | yes | yes | per-hunk gesture but accept_diff carries NO hunk id -> accepting one hunk fires whole-diff accept; a 2nd hunk re-fires the same accept_diff (duplicate); contract edit_hunk (wire.ts L82) never wired; keys a / Cmd+Enter |
| Reject hunk | `app/src/surfaces/ide/HunkReview.tsx:271` | diff-action | Flip hunk to rejected (fades card) and dispatch reject_diff(run_id,diff_id) via onAct | `reject_diff` | misleading | yes | yes | same whole-diff granularity problem; reject_diff has no per-hunk target; keys r / Cmd+Backspace/Delete |
| Select hunk (j/k, click) | `app/src/surfaces/ide/HunkReview.tsx:220` | diff-action | Move selected/active hunk (local sel) via j/ArrowDown, k/ArrowUp or click; auto-advances after accept/reject | `local-only` | appropriate-as-is | yes | yes | click target is a bare div onClick (not focusable); full keyboard drive via j/k |

### Explorer.tsx (6)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Filter / search input | `app/src/surfaces/ide/Explorer.tsx:108` | explorer-action | Debounced (140ms) query -> connector code_index.search {q,limit:40}; falls back to localSearch on empty/failure | `connector:code_index.search` | real-deep | yes | na | not an Intent, a connector call; mock code_index returns [] so mock shows local filename matches |
| Clear filter (x) | `app/src/surfaces/ide/Explorer.tsx:117` | explorer-action | setQuery('') clears the filter and restores the tree | `local-only` | appropriate-as-is | yes | na | - |
| Open file row | `app/src/surfaces/ide/Explorer.tsx:178` | explorer-action | For a file node: dispatch open_file(path) and call onOpen(path) to open the tab | `open_file` | real-deep | yes | no | row is a real <button> with roving tabindex; Enter/Space activate; tree is real fs.tree with MOCK_TREE fallback |
| Expand/collapse dir row | `app/src/surfaces/ide/Explorer.tsx:54` | explorer-action | For a dir node: toggle the collapsed map (local); also ArrowRight/ArrowLeft | `local-only` | appropriate-as-is | yes | na | - |
| Tree keyboard navigation | `app/src/surfaces/ide/Explorer.tsx:68` | explorer-action | ArrowUp/Down/Home/End move roving focus across visible rows | `local-only` | appropriate-as-is | yes | na | - |
| Open search hit | `app/src/surfaces/ide/Explorer.tsx:260` | explorer-action | Dispatch open_file(path, line) for the clicked search result | `open_file` | real-deep | yes | yes | hit exposes path, :line, and a code preview snippet (match provenance) |

### Terminal.tsx (2)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Run command (Enter) | `app/src/surfaces/ide/Terminal.tsx:82` | terminal-action | Enter splits the line into argv and dispatches run_command(argv,null); ack echoed in-buffer (runLine L148) | `run_command` | real-shallow | yes | na | dispatch + ack are real but NO live stdout stream; output only appears via the separate tool_progress echo (L116-127); pty_input/pty_resize custom names defined but never used |
| Abandon line (Ctrl+C) | `app/src/surfaces/ide/Terminal.tsx:93` | terminal-action | Ctrl+C clears the in-progress input line and reprints the prompt (local line editor) | `local-only` | appropriate-as-is | yes | na | - |

### ContextStack.tsx (11)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Snapshot state | `app/src/surfaces/ContextStack.tsx:62` | button | Claims to snapshot the RWKV state for instant resume; only pushNotice(state snapshot saved / instant resume) | `none/dead` | mock | yes | no | no intent, no connector, toast only; nothing persisted |
| Fork state | `app/src/surfaces/ContextStack.tsx:70` | button | Titled 'fork this state (memcpy)' but dispatches custom fleet_run{task:'fork from current state', n:2} + toast | `custom:fleet_run` | misleading | yes | no | wired to a real intent but fleet_run spawns 2 agents on a text task, not the constant-size state memcpy the label promises |
| Save skill | `app/src/surfaces/ContextStack.tsx:79` | button | Claims to save state as a reusable skill; only pushNotice(saved as a skill state) | `none/dead` | mock | yes | no | - |
| Load skill (per skill, x3) | `app/src/surfaces/ContextStack.tsx:115` | button | For each of 3 hardcoded SKILLS, load claims instant skill-state load; only pushNotice(loaded skill / X) | `none/dead` | mock | yes | no | SKILLS is a hardcoded const (L21: refactor-mode, test-writing, this-repo style); rendered x3, no backend |
| Switch model profile | `app/src/surfaces/ContextStack.tsx:135` | button | Whole model card is a button dispatching custom switch_profile{profile: m.model.profile} | `custom:switch_profile` | real-shallow | yes | yes | re-sends the CURRENT profile as the target, no picker/menu of alternatives; surfaces model id/arch/profile/sampling |
| Open retrieved file | `app/src/surfaces/ContextStack.tsx:167` | button | Dispatch open_file(path) for a retrieved span | `open_file` | real-deep | yes | yes | row shows path:range and relevance score |
| Pin retrieved span (toggle) | `app/src/surfaces/ContextStack.tsx:174` | toggle | Optimistic local steer overlay + dispatch custom pin_span/unpin_span{path,range} so the compiler honors it next turn | `custom:pin_span` | real-deep | yes | yes | aria-pressed toggle; overlay reconciles when next manifest arrives (state.ts) |
| Mute tool output (toggle) | `app/src/surfaces/ContextStack.tsx:204` | toggle | Toggle mute; dispatch custom pin_span/unpin_span{mute_tool: name} | `custom:pin_span` | real-shallow | yes | no | overloads pin_span/unpin_span for muting; no dedicated mute custom name; only payload key mute_tool disambiguates |
| Evict memory (toggle) | `app/src/surfaces/ContextStack.tsx:231` | toggle | Toggle evict; dispatch custom unpin_span/pin_span{evict_memory: fact} (inverted vs pin) | `custom:unpin_span` | real-shallow | yes | yes | overloads pin_span/unpin_span for eviction; no dedicated evict custom name; shows confidence score |
| Inject memory note | `app/src/surfaces/ContextStack.tsx:245` | other | NoteField input; Enter commits, sets local note overlay and dispatches custom pin_span{note} | `custom:pin_span` | real-deep | yes | no | NoteField primitive lives in contextstack/parts.tsx L188 |
| Pin dropped span (toggle) | `app/src/surfaces/ContextStack.tsx:268` | toggle | Toggle to pin a dropped span back; dispatch custom pin_span/unpin_span{title} | `custom:pin_span` | real-deep | yes | yes | shows would_be_tokens and drop reason |

### parts.tsx (1)

| Control | file:line | kind | primary action | wired_to | depth | kbd | prov | gap notes |
|---------|-----------|------|----------------|----------|-------|-----|------|-----------|
| Context Stratum: expand/collapse section | `app/src/surfaces/contextstack/parts.tsx:106` | toggle | Header button toggles the section body open/closed (local open state) | `local-only` | appropriate-as-is | yes | na | reusable primitive instantiated ~8x in ContextStack (Skills, Model, Retrieved, Tools, Memory, Dropped, Tests & state, Current action); counted once |

