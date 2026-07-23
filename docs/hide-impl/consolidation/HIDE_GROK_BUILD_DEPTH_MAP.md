# HIDE x grok-build depth map

Donor depth audit for the HIDE consolidation. Read-only pass over the donor
clone. Goal is not to copy subsystems, it is to name, per mechanism, the
smallest portable unit and how it rewrites around HIDE's spine
(protocol = hide-protocol, event store = hide-backend event log / replay.rs,
effects = hide-core/hide-security effect enum, artifacts = blob/CAS,
verify = hide-verify).

House rule for this doc: no em or en dash, hyphen and parentheses only.

## Header

- Donor repo: grok-build (SpaceXAI "grok" CLI/TUI), clone root
  `/Users/scammermike/Downloads/hide-donor-analysis/grok-build`
- Confirmed license: Apache License, Version 2.0. The LICENSE file header reads
  `Copyright 2023-2026 SpaceXAI`, but the body is the verbatim Apache-2.0 text
  (read in full). First-party code is Apache-2.0 per README ("First-party code
  in this repository is licensed under the Apache License, Version 2.0").
  Porting first-party files IS permitted, subject to Apache-2.0 section 4:
  retain the copyright and license notice, and carry a prominent "changed
  files" notice (4(b)) plus any NOTICE content.
- License caveat (gates two donor subtrees): the tool implementations under
  `crates/codegen/xai-grok-tools/src/implementations/opencode/` are ported from
  sst/opencode under the MIT license (Copyright (c) 2025 opencode), and those
  under `.../implementations/codex/` are ported from openai/codex under
  Apache-2.0 (Copyright 2025 OpenAI). These are third-party, not SpaceXAI
  first-party. See `crates/codegen/xai-grok-tools/THIRD_PARTY_NOTICES.md`. Do
  not treat those files as SpaceXAI-owned.
- HEAD commit (donor clone): `ba76b0a683fa52e4e60685017b85905451be17bc`
- Donor `SOURCE_REV` (upstream monorepo SHA):
  `ba69d70c2f7d70a130a323b2becdf137af784c7f`

---

## 1. Checkpoint store (priority: high)

- Donor location:
  `crates/codegen/xai-grok-workspace/src/session/checkpoint.rs`,
  `.../session/checkpoint_store.rs`,
  `.../session/file_state.rs` (RewindPoint, FileSnapshot).
- What it does: a `RewindCheckpoint` is keyed by `prompt_index` and bundles
  per-domain state, filesystem `RewindPoint` (before / after file snapshots),
  an optional incremental hunk delta, and (optionally) git HEAD/index. The
  `CheckpointStore` mirrors each finalized checkpoint to disk co-located in the
  session working tree (`<cwd>/.grok/rewind-checkpoints/<session_id>/checkpoint-<prompt_index>.json`),
  fronted by an in-memory `BTreeMap` cache. Retention is cap-bounded (default
  64, oldest `prompt_index` evicted). An `io_lock` serializes persist against
  truncate so disk and cache never drift. Durability is gated by an env flag
  (`GROK_WORKSPACE_REWIND_DURABLE`); the store is a durability mirror, not the
  restore path (restore is always in-process).
- Smallest portable unit: the `RewindCheckpoint` record shape plus the
  `CheckpointStore` persist / truncate-from / cap-eviction / rehydrate logic
  (roughly 200 useful lines). Additive `#[serde(default)]` domain fields are
  the pattern to keep. Drop the git/jj domain and the rootfs-snapshot coupling.
- HIDE integration point: fold the record into the hide-backend event log
  (one checkpoint entry per finalized turn, keyed by the hide-protocol turn id),
  and push the file snapshots into blob/CAS so identical content dedups instead
  of being re-serialized per turn. The BTreeMap cache + cap eviction become a
  read-through over the event log; the on-disk JSON store is redundant once the
  event log is the durable copy.
- Port classification: adapted-port. First-party Apache-2.0 code; the record
  and store logic port cleanly, the domain fan-out (git/jj/rootfs) is dropped
  and the durability target is rewritten onto the event log + CAS.

## 2. Rewind (priority: high)

- Donor location:
  `crates/codegen/xai-grok-workspace/src/session/checkpoint.rs` (TurnBoundary,
  begin/end prompt fan-out) and
  `.../session/file_state.rs` (`RewindPoint`, `rewind_files`,
  `merge_rewind_points_from`).
- What it does: restores all enabled domains together to a prompt boundary. FS
  rewind reverts in-process by writing back the before-snapshots, never via a
  rootfs rollback. `merge_rewind_points_from` folds a contiguous range of
  prompts into one target (rewinding to N collapses points N..latest):
  before-snapshots keep the earliest via `or_insert`, after-snapshots keep the
  latest. after_snapshots double as external-modification detection (if the
  file on disk differs from the recorded after-snapshot, something outside the
  agent changed it). Turn boundaries route through one internal entry point
  (`on_turn_boundary`) with a `TurnBoundary` enum distinguishing turn hooks
  from rewind RPC arms.
- Smallest portable unit: the fold/merge algorithm
  (`merge_rewind_points_from`) plus the before/after snapshot model and the
  external-modification check (roughly 150 lines). This is the valuable,
  non-obvious part; the write-back path is trivial and gets rewritten.
- HIDE integration point: hide-backend replay.rs drives it. Rewinding to turn
  T = re-materialize the before-snapshots (fetched from blob/CAS) for every
  file touched in turns >= T, expressed as a set of filesystem effects from the
  hide-core/hide-security effect enum so the revert flows through the same
  effect-application path as forward edits. The fold logic determines which
  snapshot wins when collapsing a range.
- Port classification: adapted-port. Port the fold/merge and
  external-modification logic; rewrite the apply path onto HIDE effects.

## 3. Replay (priority: medium)

- Donor location:
  `crates/codegen/xai-chat-state/src/persistence.rs` (ChatPersistence trait,
  `chat_history.jsonl`), and the "persist so replay re-applies" discipline
  threaded through
  `crates/codegen/xai-grok-shell/src/tools/notification_bridge.rs`
  (`updates.jsonl`).
- What it does: two append-only logs. `chat_history.jsonl` records one
  ConversationItem per line, with a `replace_history` operation for compaction
  and rewind. `updates.jsonl` records ACP-facing notifications (background-task
  state, task correlation, todo/loop mode, process exits) so that on
  resume/reconnect (`loadSession`) the client state is reconstructed by
  re-applying the log, not just the messages. The ChatPersistence trait
  (`persist_message` / `replace_history` / `flush`) is owned exclusively by the
  chat-state actor, so there are no locks.
- Smallest portable unit: the contract, not the code. Two rules: (a) persist
  side-effect notifications, not only assistant messages, so reconnect nets out
  correct task/loop/exit state; (b) rewind and compaction are `replace_history`,
  a truncate-and-rewrite of the log, not an in-place edit.
- HIDE integration point: hide-backend already owns `replay.rs` and an event
  log. This mechanism is the checklist of what must be in that log (tool/task
  lifecycle events, mode changes, exits) and the truncate-on-rewind rule
  (rewinding to turn T truncates the event log after T's boundary). No new
  store is needed.
- Port classification: behavioral-inspiration-only. HIDE has the store; adopt
  the log-contents discipline and the replace-history-on-rewind rule.

## 4. Hunk tracking (priority: medium; high if per-hunk accept/reject UI is a goal)

- Donor location: `crates/codegen/xai-hunk-tracker/` (dedicated crate).
  Core files: `src/diff.rs` (unified diff + hunk extraction), `src/types.rs`
  (`Hunk`, `HunkId`, `HunkSource`, `HunkTurnDelta`, `TrackingMode`),
  `src/actor/` (message-passing actor), `src/loc/` (JSONL LOC-stats sink).
- What it does: actor-based tracking of file hunks with source attribution,
  `HunkSource::AgentEdit { prompt_index }` vs `External` vs
  `ExternalEditOnAgentFile`. Diffs are computed with the `similar` crate
  (3 context lines, 10s per-diff timeout, 1 MB file cap). It records agent
  writes and watches fs_notify for external edits, emits `HunkEvent`
  (added/removed) to a client channel, and supports `HunkAction`
  (accept/reject). It produces the `HunkTurnDelta` that the checkpoint bundle
  (mechanism 1) carries.
- Smallest portable unit: `diff.rs` (unified-diff + hunk boundary extraction
  over `similar`) plus the `types.rs` attribution enums and `HunkTurnDelta`.
  That is the core; the actor + fs_notify wiring is heavier and HIDE-specific.
- HIDE integration point: attribution keys off the hide-protocol turn id. Agent
  vs external is decided by provenance, a write that arrived as a hide-tools
  effect is AgentEdit, a write seen only by a filesystem watcher is External.
  The per-turn delta rides inside the checkpoint record in the event log.
  `HunkAction` accept/reject maps onto hide-verify (accept = keep the effect,
  reject = enqueue the inverse effect).
- Port classification: adapted-port for `diff.rs` + `types.rs` (Apache-2.0,
  depends on `similar` which HIDE can pull directly);
  clean-room-reimplementation for the actor wiring around HIDE's own effect
  stream (do not port the fs_notify/actor plumbing verbatim).

## 5. Resumable subagents (priority: medium)

- Donor location:
  `crates/codegen/xai-grok-subagent-resolution/src/resume.rs`
  (`validate_resume_identity`, `ResumeValidationError`) and
  `.../src/types.rs` (`ResumeSourceData`). The crate is deliberately pure
  (no session/coordinator/transport deps).
- What it does: defines the resume contract. A resumed child inherits the
  source's raw transcript, tool state, and model (the model is pinned, any
  caller override is soft-ignored), while the system prompt and prompt context
  are freshly re-rendered from the current agent definition. `ResumeSourceData`
  carries `subagent_id`, `subagent_type`, `persona`, `model_id`, `child_cwd`,
  `worktree_path`, `snapshot_ref`, `child_session_id`.
  `validate_resume_identity` rejects a resume whose requested type or persona
  differs from the source (type checked before persona; model is not an
  identity gate).
- Smallest portable unit: `ResumeSourceData` (the resume-handle shape) plus
  `validate_resume_identity` (about 60 lines, pure, already unit-tested in the
  donor). The real value is the contract: what is inherited (transcript + tool
  state + model) vs re-rendered (prompt), and the identity gate.
- HIDE integration point: the inherited transcript + tool state are a replay of
  the child's `child_session_id` from the hide-backend event log; the
  `worktree_path` / `snapshot_ref` become a blob/CAS artifact ref (the child's
  filesystem checkpoint, mechanism 1). `validate_resume_identity` runs before
  hide-backend spawns the child program-runtime. Map `subagent_type`/`persona`
  onto HIDE's agent/role identity fields.
- Port classification: direct-licensed-port for `validate_resume_identity`
  (tiny, self-contained, Apache-2.0, with tests); adapted-port for
  `ResumeSourceData` (rename its id/path fields onto HIDE ids and CAS refs).

## 6. Terminal + stream handling (priority: high for the stream/background pattern; low-medium for full PTY)

Two distinct donor pieces solve different problems; treat them separately.

- Donor location (a), plain command streaming:
  `crates/codegen/xai-grok-tools/src/computer/local/terminal.rs`
  (actor-based `LocalTerminalBackend` / `LocalTerminalActor`),
  with `crates/codegen/xai-tty-utils/` (detach from controlling tty, suppress
  interactive pagers, process-group lifecycle). The bash tool wrapper is
  `.../xai-grok-tools/src/implementations/grok_build/bash/mod.rs`.
- What (a) does: foreground and background command execution via an actor (no
  locks, all state through channels). Output is streamed as `BashOutputChunk`
  notifications on an interval (default 100ms) and tee'd to a file for later
  read. Background runs return a `BackgroundHandle` + `TaskSnapshot`; kill
  routes through a process-group with a `KillOutcome`. Includes a cgroup memory
  monitor / OOM path (Linux, sandbox-specific).
- Donor location (b), interactive/TUI control:
  `crates/codegen/ptyctl/` (`src/session.rs`, `src/wait.rs`, `src/term.rs`,
  `src/styled.rs`), a self-contained headless PTY controller built on
  `alacritty_terminal`.
- What (b) does: spawns a process in a real PTY, sends keystrokes, reads the
  screen as text / styled / HTML, keeps scrollback, and exposes
  `WaitCondition` / `WaitOutcome` for "wait until output settles" semantics. It
  is the tool for driving interactive TUI programs, not for plain commands.
- Smallest portable unit: for plain commands, the chunked-output-to-file +
  background `TaskSnapshot` + process-group-kill pattern from (a) (minus the
  cgroup/sandbox coupling). For interactive programs, `ptyctl`'s `session.rs` +
  `wait.rs` (the output-settling `WaitCondition` is the non-obvious gem).
- HIDE integration point: stream chunks become hide-protocol tool-output events
  on the event bus (the 100ms-interval chunking maps directly to streamed tool
  output). A background `TaskSnapshot` is a resumable tool call whose state
  lives in the event log (ties into mechanism 3 replay: task correlation and
  exit persisted so reconnect nets out correct state). Kill routes through a
  hide-security process-control effect. `ptyctl`, if adopted, is a hide-tools
  backend for interactive commands only.
- Port classification: clean-room-reimplementation for the plain-command
  stream/background pattern (small, and it must be rewritten onto HIDE's event
  bus and effect enum anyway; also avoids entangling with the donor's
  cgroup/sandbox code and with the MIT opencode bash port). adapted-port for
  `ptyctl` (Apache-2.0, self-contained) only if HIDE actually needs to drive
  interactive TUIs, since it pulls the `alacritty_terminal` dependency.

## 7. ACP session behavior (priority: low)

- Donor location: `crates/codegen/xai-acp-lib/` (gateway, message typing,
  channels, line/stdin readers) plus
  `crates/codegen/xai-grok-workspace/src/file_system/acp_fs.rs`.
- What it does: a typed bidirectional gateway over the external
  `agent_client_protocol` crate (the Zed ACP standard), request/response
  correlation, side-typed messages (`AcpSide`, agent vs client), a gateway
  sender/receiver with optional tracing spans, and line-buffered stdio reading.
  It is transport + message typing; the actual session semantics (loadSession
  replay, permission prompts) live in the shell, not this lib.
- Smallest portable unit: none worth porting as code. HIDE already ships a
  `hide-acp` crate (handshake, session, tool_call, transport, unified_diff,
  permission). The donor's lib is thin glue over a third-party crate HIDE would
  pull directly.
- HIDE integration point: `hide-acp` already owns this surface. Use the donor
  as a reference for two disciplines only: side-typed channels with type
  erasure (`StorageMarker`/`Boxed`) to keep agent and client messages from
  crossing, and the loadSession-replays-updates.jsonl contract (mechanism 3).
- Port classification: behavioral-inspiration-only. HIDE has its own ACP; the
  donor lib is a wrapper over a crate we would depend on directly regardless.

---

## What NOT to import

- The MIT opencode ports
  (`xai-grok-tools/src/implementations/opencode/` -- bash, edit, glob, grep,
  read, skill, todowrite, write) and the Apache-2.0 openai/codex ports
  (`.../implementations/codex/` -- apply_patch, grep_files, list_dir,
  read_file). These are third-party with their own copyright and attribution
  obligations, not SpaceXAI first-party. Reimplement clean-room rather than
  carry two more upstreams' notices.
- The VCS domain of checkpoints:
  `xai-grok-workspace/src/session/git.rs` (about 4150 lines) and `.../jj.rs`.
  HIDE should not adopt Jujutsu, and the git HEAD/index checkpoint domain is
  optional. Keep the filesystem + event-log checkpoint domain only.
- The hub / server / daemon plumbing:
  `xai-grok-workspace/src/hub*.rs`, `.../bin/workspace_server.rs`,
  `.../daemonize.rs`, and `.../foreign_sessions/` (importers for Claude and
  Codex sessions). This is grok-specific process topology; HIDE has its own
  backend/host.
- Full terminal emulation via `alacritty_terminal` (the `ptyctl` dependency)
  unless driving interactive TUIs is an actual HIDE goal. For plain command
  execution it is overkill.
- The env-flag names, telemetry, and `xai-mixpanel` wiring. Do not copy the
  `GROK_*` flag surface; HIDE has its own config.
- The vendored `third_party/` Mermaid stack and the donor's `xai-acp-lib`
  wrapper. Where a real upstream crate exists (`agent_client_protocol`,
  `similar`, `alacritty_terminal`), depend on it directly instead of porting
  the donor's glue.

## Apache-2.0 obligation reminder (for anything actually ported)

For every first-party file or fragment ported (mechanisms 1, 2, 5, and the
optional `ptyctl` in 6): retain the `Copyright 2023-2026 SpaceXAI` notice and
the Apache-2.0 license reference, and add a prominent "this file was changed
from the SpaceXAI original" notice per Apache-2.0 section 4(b) (a
THIRD_PARTY_NOTICES entry in the HIDE tree is the clean way). Clean-room and
behavioral-inspiration items (3, 4-actor, 6-stream, 7) carry no code and no
attribution obligation, but keep the design notes traceable.
