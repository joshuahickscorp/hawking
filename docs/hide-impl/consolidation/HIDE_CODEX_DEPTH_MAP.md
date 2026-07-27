# HIDE Codex Depth Map

Donor depth audit for the HIDE consolidation. Read-only pass over an OpenAI Codex clone. This
maps the mechanisms HIDE's consolidation needs to the smallest portable unit in the donor, and
records how each would be rebuilt around HIDE's own spine (hide-protocol, the hide-core event
log, the hide-core/hide-security effect enum, blob/CAS artifacts, hide-verify).

## Header

- Repo: OpenAI Codex (donor clone at `/Users/scammermike/Downloads/hide-donor-analysis/codex`)
- Confirmed license: Apache License 2.0 (read from `LICENSE`; `NOTICE` records "OpenAI Codex,
  Copyright 2025 OpenAI" plus an MIT-licensed Ratatui derivation for the TUI). Apache-2.0 permits
  a direct licensed port with attribution and a NOTICE entry; none of the mechanisms below force
  a port, most are better rebuilt.
- Head commit: `678157acaa819d5510adfe359abb5d0392cfe461` (branch `main`)

HIDE anchor points referenced below (confirmed present in `~/Downloads/hawking-hide-build`):

- protocol: `crates/hide-protocol` (item.rs, protocol.rs, ids.rs, model.rs)
- event store: `crates/hide-core/src/event.rs` and `persistence.rs` (`InMemoryEventLog`,
  `DynEventLog`, `DynProjectionStore`), consumed across `crates/hide-backend`
- effects: `crates/hide-core/src/types.rs` (`EffectKind`, `Effect`, `EffectSet`), enforced by
  `crates/hide-security`
- artifacts: blob/CAS (digest path `crates/hide-backend/src/digest.rs`)
- verify: `crates/hide-verify` (gate.rs, oracle.rs, receipt.rs, finding.rs)
- replay / loaded-vs-stored: `crates/hide-backend/src/replay.rs`
- approvals: `crates/hide-backend/src/approval.rs`

---

## 1. Durable thread behavior

- Donor location: `codex-rs/thread-store/src/store.rs` (the `ThreadStore` trait), and
  `codex-rs/thread-store/src/live_thread.rs` (`LiveThread`, `LiveThreadInitGuard`).
- What it does: `ThreadStore` is a storage-neutral persistence boundary with an explicit
  lifecycle: `create_thread` / `resume_thread` / `append_items` / `persist_thread` (materialize
  lazy state, then flush) / `flush_thread` (make durable and readable) / `shutdown_thread` (flush,
  then close the writer) / `discard_thread` (drop the writer without forcing lazy state durable).
  `LiveThread` is the caller-facing handle that keeps lifecycle decisions above the store while
  delegating storage. `LiveThreadInitGuard` owns the live thread only while session init is still
  fallible: if init returns early, `Drop` (or an explicit `discard`) tears down the writer so a
  half-initialized session leaves no durable turd; `commit` hands ownership to the running session.
- Smallest portable unit: two things, not the trait's 20 methods. (a) The four-verb durability
  contract `persist` / `flush` / `shutdown` / `discard` with the crucial distinction that discard
  does NOT flush lazy in-memory items. (b) The `LiveThreadInitGuard` drop-discard pattern (about 40
  lines): a guard that discards persistence on early-return and is neutralized by `commit()`.
- HIDE integration point: HIDE already has the durable substrate (`hide-core` event log +
  projection store), so the store trait itself is redundant. Add the lifecycle contract as thin
  methods over `hide-backend`'s event-log append path, and add the init-guard around HIDE's
  session bring-up so a failed `initialize` (mechanism 5) discards any partial event stream.
- Port classification: adapted-port for the four-verb contract; clean-room-reimplementation for
  the init guard (it is a tiny RAII idiom, not worth vendoring).
- Priority: high.

## 2. Partial-history fork semantics

- Donor location: `codex-rs/thread-store/src/live_thread.rs`
  (`LiveThread::create_with_inherited_model_context`), `codex-rs/thread-store/src/types.rs`
  (`CreateThreadParams.forked_from_id`, `.parent_thread_id`, `.subagent_history_start_ordinal`,
  and `canonical_history_mode_from_rollout_items`).
- What it does: a fork/subagent thread is created with an inherited model-context prefix already
  written as its first durable records. Before copying the prefix, it computes
  `subagent_history_start_ordinal = persisted_prefix_len + 1`, the first ordinal that belongs to
  the child's own history. That boundary is stored in metadata BEFORE the copied prefix so history
  projection can tell inherited context apart from the child's records immediately. Forked rollouts
  keep the source `SessionMeta` after the child's own, so `canonical_history_mode_from_rollout_items`
  takes the contract from the FIRST `SessionMeta` (the child's), not any copied one.
- Smallest portable unit: the ordinal-boundary algorithm. Count the persisted prefix items, mark
  the next ordinal as the child's history start, persist that marker before the copied prefix, and
  resolve any ambiguous per-thread config from the first meta record. Roughly one function plus a
  metadata field.
- HIDE integration point: in `hide-backend`, model a fork as a new event stream that opens with a
  `ForkPoint { parent_thread, start_ordinal }` marker event, then replays the parent's model-visible
  prefix as inherited records. Projections and `hide-backend/src/replay.rs` read the marker to scope
  "the child's own turns" for paging, search (mechanism 9), and rollback.
- Port classification: clean-room-reimplementation (the value is the boundary rule, the donor code
  is welded to `RolloutItem` and the persistence policy filter).
- Priority: high.

## 3. Side-conversation lifecycle

- Donor location: `codex-rs/core/src/session/review.rs` (`spawn_review_thread`),
  `codex-rs/core/src/tasks/review.rs` (`ReviewTask`, `start_review_conversation`,
  `process_review_events`, `exit_review_mode`), spawned via
  `codex-rs/core/src/codex_delegate.rs` (`run_codex_thread_one_shot` /
  `run_codex_thread_interactive`).
- What it does: a bounded, self-contained sub-conversation (review) runs as a child session with
  its own `TurnContext`: constrained features (web search, collab, view-image disabled), a fixed
  rubric as base instructions, and `approval_policy = Never`. The parent seeds one synthesized user
  message, then consumes the child's event channel, forwarding a filtered subset (assistant deltas
  and the assistant `ItemCompleted` are suppressed in favor of structured output). The lifecycle is
  bracketed by two boundary items, `EnteredReviewMode` and `ExitedReviewMode`, so a UI can switch
  modes. On exit the summarized result is folded back into the PARENT transcript as a user message
  (the rendered exit prompt) plus an assistant message (the review output), and rollout persistence
  is explicitly materialized because a review can run before any regular user turn.
- Smallest portable unit: the boundary-and-foldback pattern, independent of "review". Emit an
  `Entered<Mode>` marker, run a child with a constrained context over a bounded event stream with a
  forwarding filter, emit an `Exited<Mode>` marker carrying the summarized result, then record a
  compact user+assistant summary back into the parent stream. The event-forwarding filter and the
  fallback JSON extraction in `parse_review_output_event` are the reusable bits.
- HIDE integration point: `hide-backend` runs the child on a child event log; the parent's
  projection appends `SideConversationEntered` / `SideConversationExited` events to its own log and
  folds the child's summary in as parent items. Constrained context maps to a scoped `EffectSet`
  (mechanism 7 anchor) for the child.
- Port classification: clean-room-reimplementation (deeply coupled to codex `Session` / `TurnContext`
  / feature flags; only the lifecycle shape transfers).
- Priority: medium.

## 4. Bounded event queues

- Donor location: `codex-rs/core/src/session/mod.rs`
  (`SUBMISSION_CHANNEL_CAPACITY: usize = 512`), `codex-rs/core/src/codex_delegate.rs`
  (`async_channel::bounded(SUBMISSION_CHANNEL_CAPACITY)` for submissions and ops), and
  `codex-rs/core/src/client.rs` (`RESPONSE_STREAM_CHANNEL_CAPACITY: usize = 1600`,
  `mpsc::channel(RESPONSE_STREAM_CHANNEL_CAPACITY)` for the model response stream).
- What it does: every hot path uses a fixed-capacity async channel. Submissions and internal ops
  ride bounded `async_channel` queues (capacity 512); the streamed model response rides a bounded
  tokio mpsc (capacity 1600). Backpressure is implicit: producers `send().await` and block when the
  queue is full, so a slow consumer throttles the producer instead of growing memory without bound.
  There is no ring buffer and no drop-oldest, it is pure await-backpressure.
- Smallest portable unit: nothing to copy. The mechanism is "use bounded channels with a named
  capacity constant and await on send". The only transferable artifacts are the two chosen numbers
  (512 submissions, 1600 response events) as sane starting sizes.
- HIDE integration point: wherever `hide-backend` bridges the model/tool producer to the UI bus
  (`ui_bus.rs`) or the event log, size those channels with named constants and rely on
  `send().await` backpressure. Do not add a bespoke queue type.
- Port classification: behavioral-inspiration-only.
- Priority: medium.

## 5. Initialization and capability negotiation

- Donor location: `codex-rs/app-server-protocol/src/protocol/v1.rs` (`InitializeParams`,
  `ClientInfo`, `InitializeCapabilities`, `InitializeResponse`).
- What it does: the first request is `initialize`. The client sends `ClientInfo { name, title,
  version }` and an optional `InitializeCapabilities`: `experimental_api` (opt into experimental
  methods/fields), `request_attestation`, `mcp_server_openai_form_elicitation`, and
  `opt_out_notification_methods` (exact wire method names the connection does not want, for example
  `thread/started`). The server replies `InitializeResponse { user_agent, codex_home,
  platform_family, platform_os }`. Capability negotiation is a plain typed struct exchange with
  serde defaults for forward compatibility, not a version handshake.
- Smallest portable unit: the request/response pair plus the capabilities struct, especially the
  two negotiation levers, `experimental_api` (a single gate that unlocks a whole method surface,
  see mechanism 6) and `opt_out_notification_methods` (per-connection notification suppression by
  method name). Both are cheap and high-value.
- HIDE integration point: add an `Initialize` method to `hide-protocol` with a
  `ClientInfo` + `ClientCapabilities` params struct and a server-info response. Store the negotiated
  capabilities per connection in `hide-backend` and consult `opt_out_notification_methods` in the
  notification emit path; gate any experimental protocol variants on `experimental_api`.
- Port classification: adapted-port (the shape is clean and directly usable; field set is trimmed
  to HIDE's needs).
- Priority: high.

## 6. Generated protocol schemas

- Donor location: `codex-rs/app-server-protocol/src/protocol/common.rs` (the
  `client_request_definitions!`, `server_request_definitions!`, `server_notification_definitions!`,
  `client_notification_definitions!` macros, about lines 194 to 1490),
  `codex-rs/app-server-protocol/src/rpc.rs` (JSON-RPC envelope types), and
  `codex-rs/app-server-protocol/src/export.rs` plus `schema_fixtures.rs` (TypeScript and JSON-Schema
  emission via `ts-rs` and `schemars`, with fixture snapshots).
- What it does: one declarative list of variants generates, per method, a method-tagged
  `ClientRequest` enum, a matching typed `ClientResponse` / `ClientResponsePayload`, method-name and
  request-id accessors, a `serialization_scope()` (thread / process / global concurrency scoping key),
  and experimental-gating metadata. `export.rs` then walks these to emit generated TypeScript and
  JSON schemas; `schema_fixtures.rs` snapshots them so protocol drift is caught in CI. The envelope
  is JSON-RPC-shaped but intentionally omits the `"jsonrpc": "2.0"` field.
- Smallest portable unit: the macro pattern, not the export binaries. A single `macro_rules!` that
  takes `Variant => "wire/name" { params, response }` rows and expands to the tagged request enum +
  typed response enum + id/method accessors. The `serialization_scope` idea (each request declares a
  concurrency key so the server can serialize same-scope work) is a second, separable idea worth
  lifting.
- HIDE integration point: define HIDE's request/notification surface in `hide-protocol` through one
  such macro so the wire enum, response typing, and a JSON-Schema export stay in lockstep from a
  single source. Reuse `schemars`/`ts-rs` only if HIDE actually ships a typed client; otherwise emit
  JSON Schema alone. Feed the per-request scope key into `hide-backend`'s request dispatch.
- Port classification: adapted-port for the macro; the `ts-rs`/`schemars` export tooling is a
  direct-licensed-port if a typed FE client is in scope, otherwise skip it.
- Priority: high.

## 7. Approval events

- Donor location: `codex-rs/protocol/src/approvals.rs` (`ExecApprovalRequestEvent`,
  `ApplyPatchApprovalRequestEvent`, `ExecApprovalRequestEvent::default_available_decisions` and
  `::effective_approval_id`, `ElicitationRequest`, guardian assessment types). Approval
  request/response params also appear in `app-server-protocol/src/protocol/v1.rs`
  (`ExecCommandApprovalParams`, `ApplyPatchApprovalParams`).
- What it does: when the agent wants to run a command or apply a patch, it emits a typed approval
  request carrying the action, cwd, turn id, an optional human reason, and context-specific
  escalation payloads (a proposed execpolicy prefix amendment, network-policy amendments, or an
  additional filesystem permission profile). `default_available_decisions` derives the ordered set
  of choices to present from that context: network context yields approve / approve-for-session /
  optional network-amendment / abort; an additional-permission ask yields approve / abort; a plain
  command yields approve / optional execpolicy-amendment / abort. `effective_approval_id` falls back
  to `call_id` for the whole-command case versus a distinct id for subcommand (execve-intercept)
  approvals. `available_decisions` is optional on the wire with a legacy fallback so old and new
  senders interoperate.
- Smallest portable unit: two pure functions and the decision enum. (a) `default_available_decisions`
  (context -> ordered decision list), and (b) `effective_approval_id` (approval-id-or-call-id
  fallback). The escalation-payload shapes (execpolicy prefix amendment, network amendment,
  additional permission profile) are the data model those functions operate over.
- HIDE integration point: HIDE already has `hide-backend/src/approval.rs` and a real effect model
  (`hide-core::types::EffectSet`). Emit an approval request event on the `hide-core` event log whose
  payload is a requested `EffectSet` delta; port `default_available_decisions` as the mapping from
  that delta to the choices a UI shows; approvals mutate the session's granted `EffectSet` enforced
  by `hide-security`.
- Port classification: clean-room-reimplementation (the decision-derivation logic is the asset; the
  surrounding codex escalation/guardian types do not match HIDE's effect enum and should not be
  copied wholesale).
- Priority: high.

## 8. Loaded-versus-stored session handling

- Donor location: `codex-rs/thread-store/src/types.rs` (`StoredThread` vs the live handle,
  `ReadThreadParams.include_history`, `ResumeThreadParams.history`, `LoadThreadHistoryParams`,
  `StoredThreadHistory`, `StoredModelContext`), and `codex-rs/thread-store/src/store.rs`
  (`read_thread`, `load_history`, `load_latest_model_context`).
- What it does: a thread has two representations. `StoredThread` is a metadata snapshot (preview,
  model, timestamps, cwd, approval mode, token usage, fork/parent linkage) optionally carrying
  history; the live handle is the active writer. Reads are explicit about cost: `read_thread` takes
  `include_history` so callers pay for replay only when needed; `load_history` returns the full
  durable replay for resume/fork/rollback, while `load_latest_model_context` returns only the suffix
  needed to reconstruct the model-visible window (stores that cannot do a targeted read fall back to
  full history). On resume, `ResumeThreadParams.history` is `Some` when the caller already has the
  replay in memory (skip the load) and `None` to load on resume, and a failed load discards the
  just-opened writer.
- Smallest portable unit: the read contract, not the SQLite store. Three levers: (a) a metadata-only
  snapshot separate from replay history, (b) an `include_history` flag on reads, and (c) the
  `load_history` (full) vs `load_latest_model_context` (targeted suffix) split, with the "resume with
  already-loaded history" fast path.
- HIDE integration point: `hide-backend/src/replay.rs` becomes the loaded-vs-stored boundary. Expose
  a metadata-snapshot projection over the `hide-core` event log distinct from a full replay, add an
  `include_history` flag on session read, and add a "latest window only" replay that stops once the
  model-visible context is reconstructed. Feed the resume fast path from an in-memory replay when the
  caller already holds it.
- Port classification: adapted-port (contract shape maps cleanly onto the event log + projections;
  drop the rollout-file and sqlite specifics).
- Priority: medium.

## 9. Transcript item paging and search

- Donor location: paging types in `codex-rs/thread-store/src/types.rs` (`ItemPage`, `TurnPage`,
  `ListItemsParams`, `ListTurnsParams`, forward/back opaque cursors). In-thread occurrence search in
  `codex-rs/thread-store/src/local/thread_history/search.rs` (`search_thread_occurrences`,
  `LiteralMatcher`, `occurrence_in_item`, the compound `SearchCursor`). Cross-thread content search in
  `codex-rs/thread-store/src/local/search_threads.rs` (ripgrep over rollout JSONL plus a
  timestamp/thread-id cursor). Backward tail paging in
  `codex-rs/rollout/src/reverse_jsonl_scanner.rs` (`ReverseJsonlScanner`).
- What it does: three capabilities. (a) Paginated reads of turns and items with forward and backward
  opaque cursors (`next_cursor` / `backwards_cursor`). (b) In-thread occurrence search: a
  case-insensitive `LiteralMatcher` produces byte ranges over the searchable text of user and final
  agent messages (agent markdown is flattened first), `occurrence_in_item` builds a bounded snippet
  with leading/trailing ellipses and projects the match range into UTF-16 code units for JS clients,
  and a compound `SearchCursor { thread_id, search_term, next_rollout_ordinal, next_occurrence_index }`
  makes pagination stable even across multiple hits inside one message. (c) Cross-thread search that
  shells out to ripgrep over the JSONL corpus, then paginates matched threads by a
  timestamp-plus-id cursor. `ReverseJsonlScanner` reads newline-delimited JSON from the end of a file
  in fixed chunks so the latest transcript tail can be paged without loading the whole file.
- Smallest portable unit: three separable pieces. (a) `LiteralMatcher::find_ranges` +
  `occurrence_in_item` (snippet windowing plus UTF-16 range projection plus the compound occurrence
  cursor), the highest-value and most storage-agnostic. (b) The forward/back opaque cursor page
  shape. (c) `ReverseJsonlScanner` (about 100 lines, pure `Read + Seek`) for backward tail paging if
  HIDE ever persists an append-only text log.
- HIDE integration point: run search over a `hide-core` event-log projection of user and agent
  messages. Port `LiteralMatcher` + snippet/UTF-16 projection directly (it is pure string work) and
  return occurrences with the compound cursor. Apply the same forward/back cursor shape to
  `hide-backend`'s item/turn read endpoints. Keep cross-thread search as an optional ripgrep pass over
  whatever durable text HIDE writes; do not adopt the SQLite occurrence index unless a store demands
  it.
- Port classification: clean-room-reimplementation for the search/snippet/cursor logic (pure and
  small, but coupled to codex item types); `ReverseJsonlScanner` is a direct-licensed-port candidate
  if HIDE needs tail paging.
- Priority: medium.

---

## What NOT to import

- The storage engines. `codex-rs/thread-store/src/local/*` and `codex-rs/rollout/*` are a large
  SQLite + JSONL persistence stack (`state_db.rs`, `session_index.rs`, `persistence_metrics.rs`,
  `sqlite_metrics.rs`, the `thread_history` sqlx schema). HIDE already has an event log and
  projection store; a second persistence engine is exactly the consolidation this effort exists to
  avoid. Take the contracts and algorithms, leave the engine.
- The `codex_protocol` type universe. `RolloutItem`, `ResponseItem`, the full `ThreadItem` taxonomy,
  `SessionMeta`, token-usage and git-info structs. These are the donor's domain model, not HIDE's;
  porting them would drag the whole crate graph and fight `hide-protocol`.
- The full generated-schema toolchain as a product. The `ts-rs` + `schemars` export binaries,
  `experimental_api` plumbing, and fixture harness are worth copying only if HIDE ships a typed
  client and a drift-checked schema. Lift the definition macro pattern first; add codegen later if
  there is a consumer.
- Codex-specific subsystems that showed up adjacent to these mechanisms: guardian assessment
  (`protocol/src/approvals.rs` guardian types), code-mode, otel/telemetry (`RolloutPersistenceTelemetry`,
  `persistence_metrics`), attestation, and the collaboration/multi-agent runtime. None are part of
  the nine target mechanisms.
- The build system. Bazel (`MODULE.bazel`, `BUILD.bazel`, `defs.bzl`) and the Nix flake are donor
  infrastructure with no bearing on the ported logic.
- No bounded-queue abstraction. Mechanism 4 is a discipline (bounded channel + await), not a type.
  Resist wrapping it.
