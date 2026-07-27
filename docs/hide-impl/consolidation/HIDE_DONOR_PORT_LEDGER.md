# HIDE Donor Port Ledger

Authoritative provenance table for the HIDE consolidation campaign. Every mechanism drawn from an inspected donor is recorded here with its chosen donor, exact donor commit, license, source files, port type, the HIDE control or crate it deepens, its status, and the notices that must ship if the port lands. If a mechanism is not in this table, it has no cleared provenance and must not be ported.

Port type vocabulary:

- `direct-licensed-port` copies donor code under its license; carries full attribution and any changed-files notice.
- `adapted-port` starts from donor code, reshapes it onto HIDE substrate; carries attribution and changed-files notice.
- `clean-room-reimplementation` rebuilds the behavior on HIDE substrate from the depth-map description; no donor code copied, so no license notice attaches, but provenance is logged here for audit.
- `behavioral-inspiration-only` borrows a discipline or sizing constant, not code; nothing to attribute.

Status vocabulary: `analyzed` (audited, cleared, slated) | `partial` (compared against the live tree; a small HIDE-side closure landed, the donor mechanism itself is still unported) | `ported` (code landed) | `improved` (landed and hardened past donor) | `deferred` (cleared but not slated this campaign). Trace E (checkpoint / rewind / fork deepening) has landed four Group A / B rows (checkpoint store, rewind, replay, partial-history fork), and Trace G (durable-thread lifecycle / Initialize / background promotion) has landed two more Group B rows (durable thread behavior, initialization and capability negotiation); Trace H (opencode head-to-head pass, 2026-07-20) moved two Group C rows to `partial`; the remaining rows are still `analyzed` or `deferred`.

## Licensing gate

The gate decides whether a donor may be code-ported at all. Result for this campaign:

| Donor | License (read from LICENSE, not assumed) | OSI-permissive | Code port permitted |
|---|---|---|---|
| OpenAI Codex | Apache License 2.0 | yes | yes, with attribution + NOTICE entry |
| grok-build (SpaceXAI grok) | Apache License 2.0 (verbatim body under a "Copyright 2023-2026 SpaceXAI" header) | yes | yes, first-party files only, with attribution + section 4(b) changed-files notice |
| opencode | MIT | yes | yes, copyright + permission notice must accompany any copied portion |

All three donors clear the gate as OSI-permissive. Because grok-build IS Apache-2.0, the conditional warning does not fire: grok-build rows keep the port types assigned in the depth map, including one `direct-licensed-port`. Had grok-build been non-permissive, every grok-build row would have been forced to `clean-room-reimplementation` or `behavioral-inspiration-only` and direct code port would have been prohibited. Recording the rule so a future license re-read can re-run the gate:

> GATE RULE: if grok-build is ever found NOT to be an OSI-permissive license, immediately downgrade all grok-build rows to `clean-room-reimplementation` or `behavioral-inspiration-only` and raise a WARNING that direct code port from grok-build is not permitted.

### WARNING: grok-build third-party subtrees are NOT first-party

grok-build's own code is Apache-2.0/SpaceXAI, but two tool-impl subtrees inside the donor clone are vendored third-party code and must not be treated as grok-build first-party:

- `xai-grok-tools/src/implementations/opencode/` is MIT, Copyright (c) 2025 opencode.
- `xai-grok-tools/src/implementations/codex/` is Apache-2.0, Copyright 2025 OpenAI.

Do not port those subtrees as grok-build code. If any HIDE need overlaps them, source it from the primary opencode or Codex donors below, under their own notices. No mechanism in this ledger draws from those subtrees.

### Clean-room rule (non-negotiable)

Cursor and Claude Code are NEVER inspected. No mechanism in this campaign may be sourced, described, or cross-checked against Cursor or Claude Code internals. They stay pure clean-room. This ledger records only the three cleared donors (Codex, grok-build, opencode); any provenance tracing to Cursor or Claude Code is a rule violation and must be rejected.

## Donor legend (full provenance)

| Donor key | Clone path | HEAD commit (full) | License | Required notice on any port |
|---|---|---|---|---|
| Codex | /Users/scammermike/Downloads/hide-donor-analysis/codex | `678157acaa819d5510adfe359abb5d0392cfe461` | Apache-2.0 | NOTICE entry: "OpenAI Codex, Copyright 2025 OpenAI"; retain Apache-2.0 header; TUI parts also carry MIT Ratatui attribution |
| grok-build | /Users/scammermike/Downloads/hide-donor-analysis/grok-build | `ba76b0a683fa52e4e60685017b85905451be17bc` (clone HEAD); upstream SOURCE_REV `ba69d70c2f7d70a130a323b2becdf137af784c7f` | Apache-2.0 | Retain "Copyright 2023-2026 SpaceXAI" notice; Apache-2.0 section 4(b) changed-files notice on every code-bearing port |
| opencode | /Users/scammermike/Downloads/hide-donor-analysis/opencode | `ba4b8e21f4ecd70f30b8dc24458f56a4fe84eca5` | MIT | MIT copyright + permission notice ("Copyright (c) 2025 opencode") must accompany any copied or substantial portion (e.g. a verbatim data table) |

Depth maps this ledger consolidates:

- docs/hide-impl/consolidation/HIDE_CODEX_DEPTH_MAP.md
- docs/hide-impl/consolidation/HIDE_GROK_BUILD_DEPTH_MAP.md
- docs/hide-impl/consolidation/HIDE_OPENCODE_DEPTH_MAP.md

---

## Group A: checkpoint / rewind / replay / hunk (grok-build + Codex)

| Mechanism | Chosen donor | Donor commit | License | Donor file(s) | Port type | HIDE integration point | Status | Required notices |
|---|---|---|---|---|---|---|---|---|
| Checkpoint store (RewindCheckpoint record + CheckpointStore persist/truncate/cap-eviction/rehydrate) | grok-build | `ba76b0a683fa` | Apache-2.0 | crates/codegen/xai-grok-workspace/src/session/checkpoint.rs, checkpoint_store.rs, file_state.rs | adapted-port | Fold the checkpoint record into hide-backend event log (one checkpoint per finalized turn, keyed by hide-protocol turn id); file snapshots to blob/CAS for dedup; BTreeMap cap-cache becomes read-through over the log | improved | SpaceXAI notice + Apache-2.0 4(b) changed-files notice (code-bearing) |
| Rewind (merge_rewind_points_from fold + before/after snapshot model + external-modification check) | grok-build | `ba76b0a683fa` | Apache-2.0 | crates/codegen/xai-grok-workspace/src/session/checkpoint.rs (TurnBoundary), file_state.rs (RewindPoint, rewind_files, merge_rewind_points_from) | adapted-port | hide-backend/src/replay.rs drives it; rewind to turn T re-materializes before-snapshots from blob/CAS as filesystem effects from the hide-core/hide-security effect enum, so revert flows the same effect-application path as forward edits | improved | SpaceXAI notice + Apache-2.0 4(b) changed-files notice (code-bearing) |
| Replay (append-only chat + side-effect notification logs; truncate-on-rewind; replace_history on compaction) | grok-build | `ba76b0a683fa` | Apache-2.0 | crates/codegen/xai-chat-state/src/persistence.rs (ChatPersistence, chat_history.jsonl), crates/codegen/xai-grok-shell/src/tools/notification_bridge.rs (updates.jsonl) | behavioral-inspiration-only | hide-backend already owns replay.rs + event log; adopt the discipline (log tool/task lifecycle, mode changes, exits; rewind to T truncates the log after T's boundary). No new store | ported | None (discipline, no code copied). Provenance logged only |
| Hunk tracking (unified-diff + hunk-boundary extraction; source attribution; HunkTurnDelta) | grok-build | `ba76b0a683fa` | Apache-2.0 | crates/codegen/xai-hunk-tracker/src/diff.rs, src/types.rs (attribution enums, HunkTurnDelta); actor + fs_notify wiring left behind | adapted-port | Attribution keys off hide-protocol turn id (write via hide-tools effect = AgentEdit; seen only by fs watcher = External); per-turn delta rides inside the checkpoint record in the event log; HunkAction accept/reject maps onto hide-verify (accept keeps effect, reject enqueues inverse effect) | analyzed | SpaceXAI notice + Apache-2.0 4(b) changed-files notice (diff.rs + types.rs are code-bearing). Note: the `similar` crate stays a normal upstream dep |
| Transcript replay tail paging (ReverseJsonlScanner, backward chunked read from file end) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/rollout/src/reverse_jsonl_scanner.rs | direct-licensed-port | The one clean direct-port candidate for Group A/B tail paging; feeds hide-backend item/turn read endpoints when HIDE writes newline-delimited durable text | analyzed | OpenAI Codex NOTICE entry + retain Apache-2.0 header (code copied) |

Notes for Group A:

- Do NOT import grok-build's git.rs/jj.rs VCS domain (roughly 4150 lines), rootfs-snapshot coupling, hub/daemon/foreign_sessions plumbing, or GROK_* flags / mixpanel telemetry. The high-value units are the record shapes and the fold/merge/eviction logic only.
- Checkpoint and Rewind share the same source files; both are cleared, but they land as two logical ports on the same donor code, so one combined 4(b) notice covers both.

## Group B: durable-thread / partial-fork / side-conversation / paging (Codex)

| Mechanism | Chosen donor | Donor commit | License | Donor file(s) | Port type | HIDE integration point | Status | Required notices |
|---|---|---|---|---|---|---|---|---|
| Durable thread behavior (four-verb persist/flush/shutdown/discard contract + LiveThreadInitGuard drop-discard RAII) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/thread-store/src/store.rs (ThreadStore trait), live_thread.rs (LiveThread, LiveThreadInitGuard) | adapted-port | Add the four-verb lifecycle as thin methods over hide-backend's event-log append path (discard must NOT flush lazy items); wrap HIDE session bring-up in the init-guard so a failed initialize discards the partial event stream | ported | OpenAI Codex NOTICE entry + Apache-2.0 header (adapted from code) |
| Partial-history fork semantics (ordinal-boundary: mark child history start before copying inherited prefix; resolve config from first meta) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/thread-store/src/live_thread.rs (create_with_inherited_model_context), src/types.rs (CreateThreadParams.forked_from_id/.parent_thread_id/.subagent_history_start_ordinal, canonical_history_mode_from_rollout_items) | clean-room-reimplementation | Model a fork as a new event stream opening with a ForkPoint { parent_thread, start_ordinal } marker, then replay the parent's model-visible prefix as inherited records; projections and hide-backend/src/replay.rs read the marker to scope the child's own turns for paging, search, rollback | ported | None (rebuilt from spec, no code copied). Provenance logged only |
| Side-conversation lifecycle (Entered/Exited boundary markers + constrained child context + forwarding filter + summary foldback) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/core/src/session/review.rs (spawn_review_thread), core/src/tasks/review.rs (ReviewTask, process_review_events, exit_review_mode), core/src/codex_delegate.rs (run_codex_thread_one_shot/_interactive) | clean-room-reimplementation | hide-backend runs the child on a child event log; parent projection appends SideConversationEntered/Exited events and folds the child summary back as parent user+assistant items; constrained context maps to a scoped EffectSet for the child | analyzed | None (rebuilt from spec). Provenance logged only |
| Bounded event queues (fixed-capacity async channels + await-backpressure; named capacity constants) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/core/src/session/mod.rs (SUBMISSION_CHANNEL_CAPACITY=512), core/src/codex_delegate.rs (async_channel::bounded), core/src/client.rs (RESPONSE_STREAM_CHANNEL_CAPACITY=1600) | behavioral-inspiration-only | At hide-backend's model/tool producer to UI bus (ui_bus.rs) or event-log boundary, size channels with named constants and rely on send().await backpressure; add no bespoke queue type. Only transferable artifacts are the two sizes (512 submissions, 1600 response events) | analyzed | None (discipline + two integers). Provenance logged only |
| Initialization and capability negotiation (typed Initialize request/response; experimental_api gate; opt_out_notification_methods) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/app-server-protocol/src/protocol/v1.rs (InitializeParams, ClientInfo, InitializeCapabilities, InitializeResponse) | adapted-port | Add an Initialize method to hide-protocol (ClientInfo + ClientCapabilities params, server-info response); store negotiated capabilities per connection in hide-backend; consult opt_out_notification_methods in the notification emit path; gate experimental protocol variants on experimental_api | ported | OpenAI Codex NOTICE entry + Apache-2.0 header (adapted from code) |
| Generated protocol schemas (client_request_definitions! macro family: tagged request enum + typed response + method/id accessors + serialization_scope key) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/app-server-protocol/src/protocol/common.rs (request/notification-definition macros, ~lines 194-1490), src/rpc.rs (JSON-RPC envelope), src/export.rs + schema_fixtures.rs (TS/JSON-Schema emission + snapshots) | adapted-port | Define HIDE's request/notification surface in hide-protocol through one macro so wire enum, response typing, and JSON-Schema export stay in lockstep; reuse schemars/ts-rs only if HIDE ships a typed client, else emit JSON Schema alone; feed the per-request scope key into hide-backend dispatch | analyzed | OpenAI Codex NOTICE entry + Apache-2.0 header (macro pattern adapted). Do NOT import the ts-rs/schemars codegen binaries wholesale |
| Approval events (default_available_decisions context to ordered decision list; effective_approval_id fallback) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/protocol/src/approvals.rs (ExecApprovalRequestEvent, ApplyPatchApprovalRequestEvent, default_available_decisions, effective_approval_id, ElicitationRequest), app-server-protocol/src/protocol/v1.rs (ExecCommandApprovalParams, ApplyPatchApprovalParams) | clean-room-reimplementation | Emit an approval-request event on the hide-core event log whose payload is a requested EffectSet delta (hide-core::types::EffectSet); reimplement default_available_decisions as the delta to UI-choices mapping; approvals mutate the session's granted EffectSet enforced by hide-security via hide-backend/src/approval.rs | analyzed | None (decision logic rebuilt from spec on HIDE's effect model). Provenance logged only |
| Loaded-versus-stored session handling (metadata snapshot vs replay; include_history flag; full vs targeted-suffix load; resume fast path) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/thread-store/src/types.rs (StoredThread, ReadThreadParams.include_history, ResumeThreadParams.history, LoadThreadHistoryParams, StoredModelContext), src/store.rs (read_thread, load_history, load_latest_model_context) | adapted-port | hide-backend/src/replay.rs becomes the loaded-vs-stored boundary: metadata-snapshot projection distinct from full replay, include_history flag on session read, latest-window-only replay that stops once model-visible context is reconstructed, resume fast path from in-memory replay when caller already holds it | analyzed | OpenAI Codex NOTICE entry + Apache-2.0 header (read contract adapted). Do NOT import the SQLite store |
| Transcript item paging and search (LiteralMatcher find_ranges + occurrence_in_item snippet/UTF-16 projection + compound cursor; forward/back opaque cursors) | Codex | `678157acaa81` | Apache-2.0 | codex-rs/thread-store/src/types.rs (ItemPage, TurnPage, cursors), src/local/thread_history/search.rs (search_thread_occurrences, LiteralMatcher, occurrence_in_item, SearchCursor), src/local/search_threads.rs (cross-thread ripgrep) | clean-room-reimplementation | Run search over a hide-core event-log projection of user + agent messages; reimplement LiteralMatcher + snippet/UTF-16 projection (pure string work) returning occurrences with the compound cursor; apply the forward/back cursor shape to hide-backend item/turn read endpoints; cross-thread ripgrep search optional | analyzed | None (pure string logic rebuilt from spec). Provenance logged only. Do NOT adopt the SQLite occurrence index |

Notes for Group B:

- ReverseJsonlScanner (the backward tail-paging scanner that also serves this group's paging) is listed once, in Group A, as the shared `direct-licensed-port`.
- What NOT to import from Codex: the thread-store/rollout SQLite+JSONL engine, the codex_protocol RolloutItem/ResponseItem type universe, guardian/otel/code-mode/attestation subsystems, and the Bazel/Nix build. HIDE's hide-core event log + projections already replace the durable substrate, so almost nothing is a wholesale port.

## Group C: command + event registry / provider-neutral boundary / agent-profile / LSP / extension-discovery / SDK-gen (opencode)

| Mechanism | Chosen donor | Donor commit | License | Donor file(s) | Port type | HIDE integration point | Status | Required notices |
|---|---|---|---|---|---|---|---|---|
| Versioned event registry + durable event store (events keyed by (type,version); manifest as wire-type to decoder map; per-aggregate latest+1 append; idempotent same-id replay; owner-claim gate) | opencode | `ba4b8e21f4ec` | MIT | packages/schema/src/event.ts (define/inventory/latest/durable/versionedType), event-manifest.ts, durable-event-manifest.ts, packages/core/src/event.ts, core/src/event/sql.ts | clean-room-reimplementation | Fold versioning + idempotent-replay rules into HIDE's existing Event/ProjectionEvent schema and replay path (hide-core DynEventLog, hide-backend/src/replay.rs); add per-aggregate sequence + owner column if absent. Do NOT import the drizzle/SQLite store or Effect PubSub | analyzed | None (discipline rebuilt from spec on HIDE's log). Provenance logged only |
| Unified command registry (name to Info catalog merging built-ins + config + MCP prompts + skills with provenance; hints() placeholder extractor) | opencode | `ba4b8e21f4ec` | MIT | packages/opencode/src/command/index.ts | clean-room-reimplementation | Complements hide-backend/src/commands.rs (intent routing) as the read side: a catalog service listing commands keyed by provenance that feeds CommandRouter on user pick; any command resolving to an effectful action must declare hide-protocol Effects so the scope check gates it | analyzed | None (rebuilt from spec). Provenance logged only |
| Provider-neutral server boundary (contract crate owns middleware placement; host injects concrete auth/effect keys; one contract backs networked + embedded transports) | opencode | `ba4b8e21f4ec` | MIT | packages/protocol/src/api.ts (makeApi/makeDefaultApi/makeApiFromGroup), packages/protocol/src/groups/*.ts, packages/server/src/api.ts + routes.ts + handlers.ts | behavioral-inspiration-only | Adopt the split in hide-protocol: contract declares middleware slots, host supplies enforcement at bind time so hide-security (sandbox, audit) + auth are injected not baked in; keep embedded-vs-networked duality but the embedded host must NEVER skip the effect/sandbox layer | analyzed | None (discipline, no code copied). Provenance logged only |
| Agent-profile configuration (Info record + mode enum + layered permission merge: built-in defaults, profile overlay, user overlay, last-match-wins) | opencode | `ba4b8e21f4ec` | MIT | packages/opencode/src/agent/agent.ts (Info schema, built-in profiles, config merge), packages/schema/src/agent.ts, packages/opencode/src/agent/subagent-permissions.ts | clean-room-reimplementation | The agent spine the consolidation needs: model the profile as a Rust struct beside hide-protocol agent ids; the per-profile ruleset compiles to hide-protocol Effect scopes enforced by hide-security (NOT to allow/deny/ask strings). Built-ins map to effect envelopes (explore = {ReadFs, Network(search)}; plan denies {WriteFs, Vcs, Process, Shell}; build = full set behind sandbox + approval gate) | partial | None (rebuilt from spec on HIDE effect scopes; nothing copied). Provenance logged only. Skip the LLM-backed generate for first port. 2026-07-20: compared against the live tree, the donor permission model is REJECTED (allow-by-default plus ask fallback) and the record is already-covered by hide-compat agents.rs; the closure that landed is HIDE-side only, Agent::allows_tool in crates/hide-compat/src/agents.rs. Built-in profiles, mode enum, and ruleset merge remain unported |
| LSP integration (static server registry Info {id, extensions, root, spawn}; NearestRoot marker-walk; extension to language-id table) | opencode | `ba4b8e21f4ec` | MIT | packages/opencode/src/lsp/server.ts (Info registry, NearestRoot/StrictNearestRoot walkers), lsp/lsp.ts (service), lsp/client.ts (JSON-RPC client), lsp/language.ts (extension to language-id map) | adapted-port | Tool-surface feature, not core spine: a future hide-lsp (or module under hide-backend/tools) holds registry + clients; spawn(root, ctx, flags) MUST route through hide-security sandbox + the hide-protocol Process effect gate (reject opencode's direct spawn with no effect check); diagnostics/symbols become UiEvents on ui_bus.rs | deferred | MIT copyright + permission notice REQUIRED for language.ts if copied verbatim (it is the one pure data table cheap to copy). JSON-RPC client is rebuilt in Rust (no notice) |
| Extension discovery (skill / plugin / MCP: convention glob scan + remote versioned staging with atomic rename; staged plan/resolve/load with typed outcomes; MCP status enum) | opencode | `ba4b8e21f4ec` | MIT | packages/opencode/src/skill/index.ts, skill/discovery.ts, plugin/loader.ts (PluginLoader plan/resolve/load), mcp/index.ts (status enum) | clean-room-reimplementation | A hide-backend discovery module feeding the command catalog (Group C mechanism 2) and agent tool sets; loading a plugin runs foreign code so it MUST cross hide-security sandbox and declare its effect envelope before any handler runs (reject opencode's dynamic-import-into-host); remote skill pull is a Network effect and writes to CAS/blob, not an ad-hoc cache dir | partial | None (rebuilt from spec; nothing copied). Provenance logged only. 2026-07-20: hide-extension-registry already exceeds all three donor loaders per capability (declared effects, scopes, sandbox, network and secret policy, provenance pinning, revocation, progressive disclosure), so the closure that landed is the missing CHECK, Registry::register now refuses Execute or Process with no sandbox isolation. Remote index pull and the dynamic-import plugin loader are REJECTED; the hide-compat to registry bridge remains unported |
| Generated SDK pattern (one contract to neutral intermediate, then per-target client emit; manifest of generated file paths; CI regenerate-and-fail) | opencode | `ba4b8e21f4ec` | MIT | packages/httpapi-codegen/src/index.ts (compile/emitEffect/emitPromise/write), packages/httpapi-codegen/README.md | behavioral-inspiration-only | hide-protocol types already derive JsonSchema; a Rust generator or thin build script can emit a typed client; adopt the manifest-tracked, CI-verified generation contract so clients never drift. Lowest urgency: HIDE is single-repo Rust, a hand-written client suffices until a second-language consumer exists | deferred | None (discipline, no code copied; the committed packages/sdk/js/src/gen output is third-party @hey-api, NOT opencode IP, so not portable). Provenance logged only |

Notes for Group C:

- Reject opencode's allow-by-default string permission model and its direct-into-host plugin/LSP code loading. Every ported effectful path (Process / Network / AgentSpawn) must cross hide-security sandbox and declare a hide-protocol Effect. This is the load-bearing HIDE improvement over the donor.
- The only verbatim-copy candidate in this group is language.ts (a factual extension to language-id map). If copied, it carries the MIT notice. Everything else is rebuilt from the depth-map description.

---

## Trace E landing (checkpoint / rewind / fork deepening)

The four rows above marked `improved` / `ported` landed on the HIDE substrate as the Trace E stage. Integration points and per-mechanism provenance:

- Checkpoint store (`improved`): `crates/hide-backend/src/services.rs` (`CheckpointRecord`) gained a `CheckpointCoverage` reference set (repo state, thread, plan, goal, artifacts; live model-state capsule stays `DEFERRED_MODEL_REQUIRED`) sealed into the blake3 integrity digest. The BTreeMap cap-cache and on-disk JSON store are NOT ported; the event log remains the durable copy. Coverage references are derived by folding the log at the boundary (`crates/hide-backend/src/host.rs`, `compute_coverage`). SpaceXAI Apache-2.0 section 4(b) changed-files notice applies once the code-bearing checkpoint/rewind ports are prepared for release.
- Rewind (`improved`): `crates/hide-backend/src/rewind.rs` (`rewind_child_events`) folds a rewind as an event-log fold, not a file write-back, and adds domain scoping (conversation / code / both) beyond the donor, plus invalidated-receipt reporting (`invalidated_receipts` reusing `hide_verify::paths_intersect`). Driven by `BackendHost::checkpoint_rewind`. Adapted from grok-build `merge_rewind_points_from`; SpaceXAI 4(b) notice on release.
- Replay (`ported`): `BackendHost::checkpoint_replay` re-applies the recorded history from the checkpoint forward onto a fresh lineage; the rewind fold IS the truncate-on-rewind discipline (the child is the folded prefix, the source is never mutated). No new store, no donor code (behavioral-inspiration-only).
- Partial-history fork (`ported`): `crates/hide-backend/src/rewind.rs` (`ForkPoint`, `start_ordinal`, `split_inherited_own`) is a clean-room reimplementation of Codex's `subagent_history_start_ordinal`; the marker is written as the child's first event by `BackendHost::checkpoint_fork` / `checkpoint_rewind` / `checkpoint_replay`, and projections/replay read it to scope inherited context from the child's own records. No donor code copied.

---

## Trace G landing (durable-thread lifecycle / Initialize / background promotion)

The two Group B rows above marked `ported` landed on the HIDE substrate as the Trace G stage (donor: Codex; campaign trace G). Model-free throughout; no model was downloaded, staged, or loaded, and no capability/quality/parity claim attaches. Integration points and per-mechanism provenance:

- Durable thread behavior (`ported`): `crates/hide-backend/src/live_thread.rs` (`LiveThread`, `LiveThreadInitGuard`) is an adapted-port of Codex `thread-store/src/store.rs` + `live_thread.rs`, reshaped onto the `hide-core` event-log append path (not the donor's 20-method `ThreadStore` trait, which the existing durable substrate makes redundant). The four verbs are thin methods over the append path: `flush` (make the buffered lazy items durable and readable), `persist` (materialize a durable `thread.persisted` marker THEN flush), `shutdown` (flush then close), and `discard` (drop the buffered items WITHOUT flushing, the crucial distinction). `LiveThreadInitGuard` is the clean-room drop-discard RAII: `Drop` discards a partial stream, `commit()` hands ownership to the running session and neutralizes the drop. Opened from the host via `BackendHost::open_live_thread`. OpenAI Codex NOTICE + Apache-2.0 header apply once the code-bearing adapted-port is prepared for release.
- Initialization and capability negotiation (`ported`): `crates/hide-backend/src/initialize.rs` (`ClientInfo`, `ClientCapabilities`, `InitializeResponse`, `ConnectionRegistry`) is an adapted-port of Codex `app-server-protocol/src/protocol/v1.rs` Initialize, trimmed to the two negotiation levers the depth map singles out: the `experimental_api` gate and `opt_out_notification_methods` (per-connection suppression by exact wire method name). `BackendHost::initialize` records the negotiated capabilities per connection; `BackendHost::notification_for_connection` (in `crates/hide-backend/src/rpc.rs`) consults them in the notification emit path, returning `None` for an opted-out method. The version-negotiation handshake already in `hide-protocol` is left untouched. OpenAI Codex NOTICE + Apache-2.0 header apply on release.
- Background promotion (campaign feature over the ported spine, not a standalone donor row): `crates/hide-backend/src/host.rs` (`promote_run_to_background`, `background_job_for_run`, `background_job_artifacts`, `resume_background_job_in_foreground`) promotes a live interactive run to a durable background job WITHOUT restarting it, by binding the running `run_id` into an additive `JobRecord.run_id` field and reusing the existing `job_create` durable path (no second store). The promoted run keeps its tokio task and its `run_id`, so it survives client disconnect; steer / pause / stop / fork reuse the existing `redirect_run` / `pause_run` / `cancel_run` / `fork_session` control intents (which already route by run id), and resume-in-foreground replays the session projection. Reachable over the intent surface via the additive custom names `promote_run` and `resume_run_foreground`. No donor code (built on HIDE's job store + interrupt hub + replay); provenance logged here for audit only.

---

## Roll-up

| Group | Mechanisms | direct-licensed-port | adapted-port | clean-room-reimplementation | behavioral-inspiration-only |
|---|---|---|---|---|---|
| A: checkpoint/rewind/replay/hunk | 5 | 1 (ReverseJsonlScanner, Codex) | 3 (checkpoint, rewind, hunk; grok-build) | 0 | 1 (replay; grok-build) |
| B: durable-thread/fork/side-conv/paging | 9 | 0 | 4 (durable thread, initialize, schemas, loaded-vs-stored) | 4 (fork, side-conversation, approvals, paging+search) | 1 (bounded queues) |
| C: registry/boundary/profile/LSP/discovery/SDK | 7 | 0 | 1 (LSP) | 4 (event registry, command registry, agent-profile, extension discovery) | 2 (provider-neutral boundary, SDK-gen) |

Code-bearing ports that carry a license notice (the only rows where donor code ships):

- grok-build checkpoint + rewind (shared source): SpaceXAI notice + Apache-2.0 4(b).
- grok-build hunk tracking (diff.rs + types.rs): SpaceXAI notice + Apache-2.0 4(b).
- Codex ReverseJsonlScanner (direct-licensed-port): OpenAI Codex NOTICE + Apache-2.0 header.
- Codex adapted-ports (durable thread, initialize, schemas, loaded-vs-stored): OpenAI Codex NOTICE + Apache-2.0 header.
- opencode LSP language.ts, only if copied verbatim: MIT copyright + permission notice.

Everything marked clean-room-reimplementation or behavioral-inspiration-only ships no donor code and carries no license notice; its provenance is logged here for audit only.

Standing reminders:

- grok-build is Apache-2.0 (gate GREEN); if that ever changes, downgrade all grok-build rows per the GATE RULE above.
- The grok-build opencode/ and codex/ tool-impl subtrees are third-party; never port them as grok-build code.
- Cursor and Claude Code stay clean-room and are never inspected. No row here traces to them.

---

## Trace H landing (opencode consolidation pass, 2026-07-20)

Five Group C mechanisms were compared head to head against the live build tree
(`docs/hide-impl/consolidation/HIDE_OPENCODE_DEPTH_MAP.md`, section "Head-to-head
comparisons"). Verdicts: agent-profile configuration and extension discovery are
`partial` (already-covered plus one small HIDE-side closure each); command and
event registry, provider-neutral boundary, and SDK generation are unchanged
(`analyzed` / `deferred`) and were document-only this pass because the command
catalog, the SDK goldens, and the frontend are frozen while a frontend pipeline
builds against them. LSP stays `deferred` and is explicitly NOT opened as a
product area.

No donor code was copied in this pass. Both closures are clean-room and
HIDE-side, so NO MIT notice attaches to either; provenance is logged here for
audit only:

- `crates/hide-compat/src/agents.rs`: added `Agent::allows_tool`, the profile
  gate predicate (deny wins, an empty allow list means inherit all). Comparing
  the donor's per-profile capability envelope exposed that HIDE's existing
  `effective_tools()` returns an empty vec for the common inherit-all shape
  (`disallowedTools` set, `tools` omitted), which a gate would read as deny-all
  or, worse, as allow-all with the deny list dropped. The donor's allow-by-
  default string ruleset, built-in profile ladder, and layered merge were
  REJECTED, not ported.
- `crates/hide-extension-registry/src/registry.rs`: `Registry::register` now
  refuses a manifest that declares `Effect::Execute` or `Effect::Process` while
  requiring no sandbox isolation (`InvalidManifest`). This makes the campaign's
  standing rule ("every ported effectful path must cross hide-security sandbox")
  a checked invariant at the one gate every capability crosses, instead of a
  convention held only by `builtin_tools.rs`. It is the direct counter to the
  donor's dynamic-import-into-host plugin loading and unchecked LSP spawn.

Verification: `cargo test -p hide-compat -p hide-extension-registry
--no-default-features` green (hide-compat 28 unit + 11 integration;
hide-extension-registry 17 unit + 15 integration + 1 doc), `cargo build
--workspace --no-default-features` clean.
