# HIDE opencode Depth Map

Donor depth audit for the HIDE consolidation. READ ONLY over the donor clone.
Goal is NOT to copy subsystems: for each mechanism this records the smallest
portable unit and how it rewrites around the existing HIDE spine (hide-protocol,
hide-backend event log, hide-security effect enum plus sandbox, blob/CAS,
hide-verify). Every ported mechanism must call HIDE effect enforcement plus
sandbox; none may adopt opencode's weaker permission defaults (see the final
note).

## Header

- Repo: opencode (donor clone at `/Users/scammermike/Downloads/hide-donor-analysis/opencode`)
- Upstream: `github/opencode` monorepo (bun workspaces, Effect + TypeScript)
- License (read from the donor `LICENSE` file, not assumed): MIT License,
  Copyright (c) 2025 opencode. MIT permits use, modification, and redistribution
  with attribution; the copyright and permission notice must travel with any
  copied or substantial portion. Ports below that copy verbatim data must carry
  that notice.
- Head commit: `ba4b8e21f4ecd70f30b8dc24458f56a4fe84eca5`
  ("feat(app): toggle debug tools from dev badge (#36689)", 2026-07-20)
- Donor layout note: two source trees coexist. `packages/opencode/src` is the
  older instance layer; `packages/core/src` plus `packages/schema/src` plus
  `packages/protocol/src` are the newer Effect-based rewrite. The rewrite carries
  the strongest portable patterns and is the primary source below.

## HIDE integration anchors (confirmed present in the build tree)

- Effect enum: `crates/hide-protocol/src/plan.rs` `pub enum Effect` (ReadFs,
  WriteFs, Network, Process, Shell, Vcs, Environment, Approval, AgentSpawn,
  State, Other), "a step or agent lists the effects it may cause so a scope
  check can gate it before execution".
- Event log plus replay: `hide-core` `DynEventLog` / `Event` / `ProjectionEvent`,
  driven through `crates/hide-backend/src/replay.rs` (projection engine plus
  bounded transcript search).
- Command router: `crates/hide-backend/src/commands.rs` `CommandRouter` (typed
  Intent in, validate, append `user.intent.*` event, return IntentAck).
- Push event surface: `crates/hide-backend/src/ui_bus.rs` `UiEventBus` (tokio
  broadcast, coalescing, bounded backpressure).
- Sandbox plus audit: `crates/hide-security/src/sandbox.rs`, `audit.rs`.
- Verify gate: `crates/hide-verify` (gate, receipt, oracle, review).

---

## Mechanism 1: Versioned event registry plus durable event store

- Donor location:
  - `packages/schema/src/event.ts` (define, inventory, latest, durable,
    versionedType, Definition/Payload types)
  - `packages/schema/src/event-manifest.ts` and
    `packages/schema/src/durable-event-manifest.ts` (manifest assembly)
  - `packages/core/src/event.ts` (the event-sourcing engine)
  - `packages/core/src/event/sql.ts` (the two-table store)
  - `packages/core/src/database/migration/20260323234822_events.ts`
- What it does: Each event type is declared once as a schema-carrying
  `Definition` with optional `durable: { version, aggregate }`. `inventory(...)`
  freezes definition arrays into a manifest, `latest()` projects highest version
  per type, `durable()` builds a `type.version -> definition` map for
  deserialization. The store (`event` plus `event_sequence` tables) enforces a
  monotonic per-aggregate sequence: publish computes `latest + 1`, replay
  requires exact `latest + 1`, and a re-published id/type/data at an existing seq
  is treated as idempotent (compared byte-wise, diverging payloads die). Ownership
  claims (`owner_id`) gate replay so a foreign writer cannot advance another
  owner's aggregate. Local projections commit atomically inside the same
  transaction as the durable insert.
- Smallest portable unit: the discipline, not the Effect engine. Three rules plus
  one registry shape: (1) events are versioned by `(type, version)` and the
  manifest is the authority that maps a wire type to a decoder; (2) durable
  events carry `(aggregateID, seq)` with strict `latest + 1` append and
  idempotent same-id replay; (3) owner claim gates cross-writer append. The
  `event.ts` `versionedType`, `latest`, and `durable` functions are ~40 lines of
  transferable logic.
- HIDE integration point: HIDE already owns the event log
  (`hide-core::DynEventLog`, `hide-backend/src/replay.rs`). Fold the versioning
  and idempotent-replay rules into the existing `Event`/`ProjectionEvent` schema
  and the replay path; add a per-aggregate sequence plus owner column if the
  store lacks one. Do NOT import the SQLite/drizzle store or the Effect PubSub;
  HIDE has its own persistence and `UiEventBus`.
- Port classification: clean-room-reimplementation (rules and manifest shape
  rebuilt in Rust against the existing log; the Effect engine is behavioral
  inspiration only).
- Priority: high.

## Mechanism 2: Unified command registry (config plus MCP plus skill sources)

- Donor location: `packages/opencode/src/command/index.ts`
- What it does: Builds one `name -> Info` catalog of runnable slash-commands by
  merging four sources into a single namespace: built-in commands (init, review),
  user config commands, MCP prompts (lazy-resolved templates), and skills. Each
  entry carries `{ name, description, agent?, model?, source, template, subtask?,
  hints[] }`. `hints()` extracts `$1..$n` and `$ARGUMENTS` placeholders from the
  template so a UI can prompt for arguments. `source` records provenance
  ("command" | "mcp" | "skill").
- Smallest portable unit: the `Info` record shape plus the merge-with-provenance
  pattern plus the `hints()` placeholder extractor (a one-function regex over
  `$\d+` and `$ARGUMENTS`). The lazy MCP template resolution is an optimization,
  not core.
- HIDE integration point: `crates/hide-backend/src/commands.rs` already routes
  typed Intents. This is the complementary read side: a catalog service that
  lists available commands for the UI, keyed by provenance, feeding the
  CommandRouter when a user picks one. Add it beside `CommandRouter`, sourced from
  HIDE config plus the skill/MCP discovery in Mechanism 6. Every command that
  resolves to an effectful action must declare its effects (Mechanism-1 events,
  hide-protocol `Effect`) so the scope check gates it.
- Port classification: clean-room-reimplementation.
- Priority: medium.

## Mechanism 3: Provider-neutral server boundary (contract owns middleware, host injects keys)

- Donor location:
  - `packages/protocol/src/api.ts` (`makeApi`, `makeDefaultApi`,
    `makeApiFromGroup`)
  - `packages/protocol/src/groups/*.ts` (one file per domain group, e.g.
    `provider.ts`, `session.ts`)
  - `packages/server/src/api.ts`, `routes.ts`, `handlers.ts`,
    `handlers/provider.ts`
- What it does: The protocol package declares a single `HttpApi` contract as a
  set of domain groups plus endpoint schemas, and owns middleware placement
  (Authorization, SchemaError, and per-group location middleware). The concrete
  middleware service keys are injected by the caller
  (`makeDefaultApi({ locationMiddleware, sessionLocationMiddleware })`), so the
  contract stays free of core service identities. The server package binds the
  same contract twice: `createRoutes` (networked, password auth) and
  `createEmbeddedRoutes` (in-process, no password). Handlers are Effect layers
  grouped by domain and merged. Net effect: one authoritative contract, two
  transports (network and embedded), provider-neutral because the contract never
  names a concrete provider or transport.
- Smallest portable unit: the boundary discipline, (1) the contract crate is the
  single authority and lists endpoints/groups plus declares where middleware
  sits, (2) the host injects concrete auth/location enforcement rather than the
  contract hard-coding it, (3) the identical contract backs both a networked and
  an embedded host. Roughly the shape of `makeApiFromGroup` plus the two
  `createRoutes` variants.
- HIDE integration point: `crates/hide-protocol` is already the contract crate.
  Adopt the "contract declares middleware slots, host supplies enforcement"
  split so hide-security (sandbox, audit) and any auth layer are injected by the
  host at bind time, not baked into the protocol. Keep the embedded-vs-networked
  duality: the embedded host reuses the same protocol as any future networked
  server, differing only in the injected auth/effect layer. Do NOT weaken this by
  letting an embedded host skip the effect/sandbox layer (opencode's embedded
  variant drops the password, but must still run the effect gate).
- Port classification: behavioral-inspiration-only (Rust plus hide-protocol
  already exist; this transfers the middleware-injection and dual-transport
  architecture, not code).
- Priority: medium-high.

## Mechanism 4: Agent-profile configuration

- Donor location:
  - `packages/opencode/src/agent/agent.ts` (Info schema, built-in profiles,
    config merge, `generate`)
  - `packages/schema/src/agent.ts` (the wire schema)
  - `packages/opencode/src/agent/subagent-permissions.ts`
- What it does: An agent profile is `{ name, description?, mode
  (subagent|primary|all), model?, variant?, prompt?, temperature?, topP?, steps?,
  options, permission: Ruleset, native?, hidden?, color? }`. A fixed set of
  built-in profiles (build, plan, general, explore, compaction, title, summary)
  is defined with per-profile permission rulesets, then user config is merged
  over each (`Permission.merge(defaults, profileRules, userRules)`), and unknown
  config keys create new profiles. The plan profile denies all edit tools except
  a plan-notes glob; explore denies everything except read-only search tools.
  Each profile thus carries its own capability envelope.
- Smallest portable unit: the `Info` record plus the layered permission-merge
  (built-in defaults, then profile overlay, then user overlay, last-match-wins)
  plus the mode enum. The LLM-backed `generate` (author a new agent from a
  description) is a feature, not core, skip it for the first port.
- HIDE integration point: this is the agent spine the consolidation needs. Model
  the profile as a Rust struct beside `hide-protocol` agent ids; the per-profile
  permission ruleset must compile to hide-protocol `Effect` scopes enforced by
  hide-security, not to opencode's `allow|deny|ask` string rules. Built-in
  profiles (a read-only explore, a no-edit plan, a full build) map onto HIDE
  effect envelopes: explore = {ReadFs, Network(search)} only; plan = deny
  {WriteFs, Vcs, Process, Shell}; build = full set behind the sandbox plus
  approval gate.
- Port classification: clean-room-reimplementation (the record and merge
  semantics rebuilt so the capability envelope is HIDE effects, not opencode
  permission strings).
- Priority: high.

## Mechanism 5: LSP integration

- Donor location:
  - `packages/opencode/src/lsp/server.ts` (the `Info` registry: `{ id,
    extensions[], global?, root: RootFunction, spawn(root, ctx, flags) }`, plus
    `NearestRoot` / `StrictNearestRoot` root-marker walkers)
  - `packages/opencode/src/lsp/lsp.ts` (the service: open documents, diagnostics,
    document/workspace symbols, publishes `LspEvent`)
  - `packages/opencode/src/lsp/client.ts` (JSON-RPC LSP client)
  - `packages/opencode/src/lsp/language.ts` (pure extension -> language-id map)
- What it does: A static registry of language-server definitions. Each entry
  detects a project root by walking up for marker files (e.g. deno.json,
  tsconfig.json), spawns or on-demand-installs the server binary, and exposes it
  as an LSP client. The service maps file extensions to servers, opens touched
  files, and surfaces diagnostics and symbols as events.
- Smallest portable unit: the registry entry shape (`Info`) plus the
  `NearestRoot` marker-walk plus the `language.ts` extension table. The JSON-RPC
  client is standard and rebuilt in Rust (or a crate). `language.ts` is a pure
  factual data map, the one piece cheap to copy verbatim (carry the MIT notice).
- HIDE integration point: LSP is a tool-surface feature, not the core spine. A
  future `hide-lsp` (or a module under hide-backend/tools) holds the registry and
  clients. The `spawn(root, ctx, flags)` step launches a subprocess, so it MUST
  route through hide-security sandbox plus the hide-protocol `Process` effect gate
  (opencode spawns servers directly with no effect check; that is exactly the
  weaker assumption to reject). Diagnostics/symbols become UiEvents on the
  `UiEventBus`.
- Port classification: adapted-port (registry shape and root-walk adapted to
  Rust; `language.ts` table is a direct-licensed-port with attribution; JSON-RPC
  client is clean-room).
- Priority: low.

## Mechanism 6: Extension discovery (skill, plugin, MCP)

- Donor location:
  - `packages/opencode/src/skill/index.ts` (local glob scan of `**/SKILL.md`
    across `.claude`, `.agents`, `.opencode`, with frontmatter parse plus
    duplicate-name warnings) and `packages/opencode/src/skill/discovery.ts`
    (remote index pull: fetch `index.json`, download files to cache, versioned
    staging plus atomic rename)
  - `packages/opencode/src/plugin/loader.ts` (`PluginLoader`: a staged
    plan -> resolve -> load pipeline with typed `Resolved` / `Missing` / `Loaded`
    outcomes and per-stage error reporting; retries only pre-import file-plugin
    setup failures)
  - `packages/opencode/src/mcp/index.ts` (config-driven local vs remote MCP
    connect, per-server status enum: connected, disabled, failed, needs_auth,
    needs_client_registration)
- What it does: Three parallel discovery loaders that turn on-disk or configured
  declarations into live extensions. Skills are discovered by directory
  convention (a `SKILL.md` with name/description frontmatter) or pulled from a
  remote index with content-addressed, version-guarded caching. Plugins are
  resolved through explicit stages so the exact skip reason (install, entry,
  compatibility, load) is reportable, and file plugins get one retry after
  dependency prep. MCP servers are connected from config by type with a typed
  status lifecycle.
- Smallest portable unit: the staged-resolution pattern with typed outcomes (the
  `PluginLoader` plan/resolve/load enum plus per-stage report callback) and the
  convention-scan-plus-remote-index pair for skills (glob for a marker file,
  frontmatter for metadata, versioned staging directory with atomic rename on
  update). The MCP status enum is a small transferable state machine.
- HIDE integration point: a `hide-backend` discovery module feeding the command
  catalog (Mechanism 2) and agent tool sets. Loading a plugin imports and runs
  foreign code, so it MUST cross the hide-security sandbox and declare its effect
  envelope before any handler runs, opencode imports plugin modules directly into
  the host process with no sandbox, which is the weaker assumption to reject.
  Remote skill pull performs Network effect (gate it) and writes to CAS/blob (use
  hide's content store rather than an ad-hoc cache dir).
- Port classification: clean-room-reimplementation (staged loader and scan
  patterns rebuilt in Rust with the sandbox in the load path).
- Priority: medium.

## Mechanism 7: Generated SDK pattern (contract-derived clients)

- Donor location:
  - `packages/httpapi-codegen/src/index.ts` (phases `compile(Api)`,
    `emitEffect(contract)`, `emitPromise(contract)`, `write(output, directory)`,
    plus the legacy one-shot `generate`)
  - `packages/httpapi-codegen/README.md` (the settled rules)
  - generated output committed at `packages/sdk/js/src/gen/*` (note: the shipped
    JS SDK under `gen/` is actually `@hey-api/openapi-ts` output from
    `openapi.json`, third-party, not opencode IP; the portable pattern is the
    `httpapi-codegen` generator itself)
- What it does: Reflects one authoritative `HttpApi` contract into a shared
  intermediate (`compile`), then emits two independent clients from it: a rich
  Effect client (decoded native values, runtime schemas, streams as `Stream`) and
  a zero-Effect Promise client (wire-oriented values, plain fetch, streams as
  `AsyncIterable`). Input fields from path/query/header/body are flattened into
  one object; duplicate field names are rejected; single `{ data: A }` success
  envelopes are unwrapped; transport/decode/status failures collapse to one
  stable `ClientError`. Generated files are tracked in `.httpapi-codegen.json` so
  regeneration removes only its own stale files, and CI regenerates then fails if
  the worktree changed (generated source is committed for review).
- Smallest portable unit: the pipeline discipline, (1) one contract is the single
  source, (2) `compile` to a neutral intermediate then `emit` per target so each
  client has its own type projection, (3) a manifest of generated file paths so
  regeneration is a clean diff, (4) CI regenerate-and-fail to keep committed
  output honest. The Effect/Promise specifics are TypeScript-only.
- HIDE integration point: `hide-protocol` types already derive `JsonSchema`. A
  Rust generator (or a thin build script over the derived schema) can emit a
  typed client for any embedding surface. Adopt the manifest-tracked,
  CI-verified generation contract so generated clients never drift from
  hide-protocol. Lowest urgency: HIDE is single-repo Rust, so a hand-written
  client against hide-protocol is fine until a second-language consumer exists.
- Port classification: behavioral-inspiration-only.
- Priority: low.

---

## What NOT to import

- Permission defaults. opencode's model (`packages/opencode/src/permission/index.ts`,
  `packages/schema/src/permission.ts`) is a last-match-wins string ruleset over
  three effects (allow, deny, ask) with `ask` as the fallback, and the build
  agent ships `"*": "allow"`. Reject the allow-by-default posture and the
  string-keyed permissions. HIDE gates on the typed hide-protocol `Effect` enum
  through hide-security sandbox plus approval; port the layered-merge shape
  (Mechanism 4) but never the permissive defaults or the "ask" fallback that can
  silently pass.
- Direct-into-host code loading. Both plugin load
  (`packages/opencode/src/plugin/loader.ts`, dynamic `import(entry)`) and LSP
  spawn (`packages/opencode/src/lsp/server.ts`, direct subprocess) run foreign
  code with no sandbox and no declared effects. Any HIDE port of Mechanisms 5 and
  6 must cross hide-security sandbox and declare a Process/Network/etc. effect
  envelope first.
- The Effect runtime and the drizzle/SQLite store. The event engine's value is
  its rules (Mechanism 1), not its `PubSub`, `Layer`, `Context.Service`, or
  drizzle tables. HIDE has its own event log, projection engine, and
  `UiEventBus`; importing the Effect/DB machinery would fork the persistence
  layer for no gain.
- The generated JS SDK under `packages/sdk/js/src/gen`. That output is
  `@hey-api/openapi-ts` (third-party) generated from `openapi.json`, not an
  opencode mechanism, and pulls a JS/TS toolchain HIDE does not want. Take the
  generator discipline (Mechanism 7), not the output.
- The whole-server subsystem. `packages/server` and
  `packages/opencode/src/server/routes/instance/httpapi/*` are a large
  Effect-HttpApi implementation. Import the boundary discipline (Mechanism 3),
  not the routes, handlers, or middleware code, which are Effect-specific and
  duplicate what hide-protocol plus hide-backend already provide.

---

# Head-to-head comparisons (consolidation pass, 2026-07-20)

The sections above are a donor audit. This section is the decision pass: five
mechanisms compared against what the HIDE build tree ACTUALLY contains today,
each with a verdict of adopt, already-covered, or reject. Two small closures
landed (comparisons 1 and 2); comparisons 3, 4, and 5 are document-only because
the command catalog, the SDK goldens, and the frontend are frozen while a
frontend pipeline runs against them.

Scope fence honored: no edit to `crates/hide-protocol/src/command.rs`,
`crates/hide-sdk/goldens/**`, `app/**`, or `pnpm-lock.yaml`; no catalog entry
added or changed; no model downloaded, staged, or selected.

## Comparison 1: Agent-profile configuration

- What opencode does: `packages/schema/src/agent.ts` defines the profile record
  `{ id, model, request, system, description, mode (subagent|primary|all),
  hidden, color, steps, permissions: Ruleset }`.
  `packages/opencode/src/agent/agent.ts` builds seven built-in profiles (build,
  plan, general, explore, compaction, title, summary) by layering
  `Permission.merge(defaults, profileRules, userRules)`. The `defaults` block
  starts at `"*": "allow"` with a few `ask` / `deny` exceptions (doom_loop,
  question, plan_enter, plan_exit, `*.env` reads). The plan profile denies
  `edit: "*"` except a plans glob; unknown `cfg.agent` keys mint new profiles.
  Permissions are last-match-wins glob strings over allow, deny, ask.
- What HIDE does: `crates/hide-compat/src/agents.rs` parses a RICHER profile out
  of `.claude/agents/*.md` frontmatter (name, description, tools,
  disallowedTools, model with an `inherit` sentinel, skills, mcp, hooks, memory,
  permissions, body), discovered user-then-project so the project wins, sorted by
  name for determinism. The capability envelope is NOT carried by the profile:
  `crates/hide-extension-registry` declares per-capability effects, scopes,
  sandbox requirement, network policy, and secret policy, and
  `crates/hide-backend/src/policy.rs` derives a typed `PolicyDecision` per CALL
  from those declared effects plus the hide-security permission verdict
  (fail-closed on Deny, Execute or Process to RequireSandbox, Irreversible or
  Privileged or GitMutation to RequireReviewer, Write follows the engine).
  `crates/hide-kernel/src/subagent/mod.rs` carries `SubagentKind`,
  `IsolationMode` (None, Context, Worktree, FreshContext, MicroVm), and a derived
  child budget.
- Verdict: ALREADY-COVERED for the profile record and REJECT for the permission
  model, plus one small ADOPT that the comparison exposed on the HIDE side.
- Why: HIDE's gate is per-call and typed against declared effects, which is
  strictly stronger than a string ruleset frozen at profile-build time, and
  importing the donor defaults would import `"*": "allow"` plus the `ask`
  fallback that this campaign explicitly rejects. The built-in profile ladder is
  not worth rebuilding: HIDE has no built-in profiles today and the equivalent
  envelopes (explore is read-only, plan denies writes, build is full behind
  sandbox plus approval) already express themselves as effect sets in policy.rs.
  The LLM-backed `generate` is model-dependent and out of this pass entirely.
  The one REAL hole was in HIDE's own code, not in the donor gap:
  `Agent::effective_tools()` returns `tools` minus `disallowedTools`, so a
  profile that sets ONLY `disallowedTools` (the common inherit-all shape, since
  omitting `tools:` means the agent keeps every tool) returns an EMPTY vec. Any
  gate reading that vec either denies everything or, reading empty as
  inherit-all, silently drops the deny list. Landed: `Agent::allows_tool(tool)`,
  the gate predicate, where deny wins and an empty allow list means inherit all,
  with `effective_tools` left alone and documented as the explicit-set question.
- Not landed, on purpose: a `mode` enum (HIDE already distinguishes subagent
  shape through `SubagentKind` plus `IsolationMode`), built-in profiles, and the
  ruleset merge. Wiring a profile into turn execution (nothing consumes
  `hide_compat::agents` today) is a build item, not a small addition, and it
  would need a command surface, which is frozen this pass.

## Comparison 2: Extension discovery

- What opencode does: three parallel loaders.
  `packages/opencode/src/skill/index.ts` globs `**/SKILL.md`,
  `{skill,skills}/**/SKILL.md`, and `skills/**/SKILL.md` across global and
  project directories, parses frontmatter, and logs a warning on a duplicate
  skill name. `skill/discovery.ts` pulls a remote `index.json`, downloads files
  to a versioned staging directory, and atomically renames.
  `plugin/loader.ts` is a staged plan, resolve, load pipeline with typed
  `Resolved` / `Missing` / `Loaded` outcomes, a per-stage report callback, and one
  retry for file-plugin dependency prep, then `import(entry)` straight into the
  host process. `mcp/index.ts` connects configured servers with a five-state
  status union (connected, disabled, failed, needs_auth,
  needs_client_registration).
- What HIDE does: discovery lives in hide-compat and is model-free and
  network-free. `skills.rs` walks `.claude/skills/**/SKILL.md` and reads
  allowed-tools, disable-model-invocation, user-invocable, context, model,
  effort, and paths; `mcp.rs` reads `.mcp.json`; `agents.rs` is comparison 1.
  The capability ABI is `hide-extension-registry`: one `CapabilityManifest` for
  any of twelve kinds, carrying declared effects with an undeclared-effects
  registration error, scope coverage matching, `SandboxReq`, `NetworkPolicy`
  denied by default, `SecretPolicy` none by default, `ContextCost`, and
  `Provenance` with version plus commit pinning and revocation. Progressive
  disclosure is proven, not asserted: a monotonic schema-load counter shows that
  registration, indexing, and resolution never materialize a schema.
- Verdict: ALREADY-COVERED for the manifest and metadata story, REJECT the remote
  skill index and the dynamic-import plugin loader, plus one small ADOPT (an
  invariant, not a module).
- Why: HIDE's registry is strictly stronger per capability than any of the three
  donor loaders, which each carry ad hoc metadata and no effect declaration at
  all. So the gap is not a missing loader, it is a missing CHECK. The registry
  trusted `SandboxReq`: `builtin_tools.rs` sets `Subprocess` for Execute or
  Process by convention, but nothing stopped a manifest from declaring
  `Effect::Process` with `SandboxReq::None` and registering successfully. That is
  exactly the donor's weak assumption (opencode imports plugin modules into the
  host and spawns language servers with no effect check) arriving through the
  back door. Landed: `Registry::register` now rejects a manifest that declares
  Execute or Process while requiring no isolation, as `InvalidManifest`. Every
  capability crosses `register`, including any future manifest minted from an
  on-disk or foreign declaration, so the sandbox rule is now checked rather than
  conventional.
- Not landed, on purpose: a hide-compat to registry bridge (turning a discovered
  skill or MCP server into a manifest with derived effects and a file-path
  provenance) is the honest next step, but it is a wiring build item with no live
  caller in the tree today, and adding an unwired module is the failure mode this
  campaign is correcting. Remote skill pull is a Network effect and stays out of
  a model-free, network-free pass. The MCP status union is a lifecycle nicety
  worth having once a live connect path exists.

## Comparison 3: Command and event registry (document only)

- What opencode does: `packages/opencode/src/command/index.ts` merges four
  sources into one `name -> Info` map: built-ins, config commands, MCP prompts
  (lazily resolved templates), and skills. `Info` is
  `{ name, description?, agent?, model?, source, template, subtask?, hints[] }`,
  where `source` records provenance as "command", "mcp", or "skill", and
  `hints(template)` extracts `$1..$n` plus `$ARGUMENTS` so a UI can prompt for
  arguments. Events are a separate registry: `packages/schema/src/event.ts`
  declares each event once with a version and an optional
  `durable: { version, aggregate }`, `inventory()` freezes the manifest, and
  `latest()` / `durable()` build the wire-type-to-decoder maps.
- What HIDE does: `crates/hide-protocol/src/command.rs` is ONE static
  `CommandSpec` table (42 commands) with far more per-command structure than
  `Info`: category, primary surface plus every available surface, required
  selection, required server capabilities, declared effects (the protocol
  `Effect` enum), approval policy, keyboard shortcut, palette and context-menu
  flags, toolbar binding, a typed `BackendBinding` (Intent, Custom, Rpc,
  LocalOnly), undo strategy, receipt kind, and telemetry name. In-crate tests
  assert unique ids, that every binding resolves to a real Intent name, a live or
  pending custom name, or a real Method or host capability, that every command
  has a shortcut or a palette entry, and that no catalog string carries an en or
  em dash. `crates/hide-sdk/src/command.rs` projects the table into
  `goldens/command_catalog.json` and `goldens/commands.d.ts`, byte-compared by
  `tests/golden.rs`, which is the frontend drift guard.
- What HIDE does better: declared effects, approval policy, and undo strategy per
  command (donor `Info` has none of the three); typed backend bindings validated
  against the real Intent and Method sets, so an invented binding fails the build;
  one generated frontend artifact the FE cannot fork by hand; and a static table
  that cannot fail to assemble at runtime.
- What opencode does better: (a) provenance-tagged DYNAMIC sources, so a command
  can arrive from user config, an MCP prompt, or a skill without a code change,
  while HIDE's table is compile-time only; (b) `hints()`, so a command declares
  its argument placeholders and a palette can prompt for them, where HIDE's
  `RequiredSelection` covers a selection but not free-form arguments; (c)
  versioned event definitions with a manifest mapping `(type, version)` to a
  decoder, which HIDE's event log does not carry.
- Worth adopting later, not now: a READ-SIDE catalog merge that unions the static
  hide-protocol table with discovered user, skill, or MCP commands, each tagged
  with its provenance and each required to declare effects before any surface may
  offer it. That is a catalog change and the catalog is frozen this pass, so
  nothing was touched. The event-versioning rule is already tracked as Group C
  mechanism 1 in the ledger and is unaffected by this pass.

## Comparison 4: Provider-neutral client and server boundary (document only)

- What opencode does: `packages/protocol/src/api.ts` declares one `HttpApi` as a
  set of domain groups. `makeApiFromGroup` places the Authorization and
  SchemaError middleware and attaches location / session-location middleware per
  group, but the concrete middleware service KEYS are parameters the host injects
  (`makeApi({ definitions, locationMiddleware, sessionLocationMiddleware })`), so
  the contract never names a core service identity. `packages/server/src/routes.ts`
  binds the same contract twice: `createRoutes(password)` and
  `createEmbeddedRoutes()`, differing only in the injected `ServerAuth.Config`
  layer. Provider neutrality comes from the contract naming no provider and no
  transport.
- What HIDE does: `crates/hide-protocol` is already the contract crate, and
  `crates/hide-serve/src/lib.rs` is a thin axum router over `BackendHost` with
  four routes (`/v1/hide/intent`, `/v1/hide/events` serving both the WS upgrade
  and the `after_seq` catch-up, `/v1/hide/connector`, `/v1/hide/rpc`) plus
  `/healthz`.
- Is HIDE genuinely provider-neutral: YES, and structurally more so than the
  donor. hide-backend has NO Rust dependency on the model runtime; it is
  HTTP-coupled to it (recorded in the workspace manifest: "HTTP-coupled to the
  runtime (no Rust dep on hawking-core/serve)"), so no provider or runtime type
  can leak into the contract by construction, not merely by discipline.
- Concrete weaknesses: (a) there is only ONE bind. `hide_serve::router(host)`
  hardcodes its own middleware, a CORS layer over a fixed origin list with a
  `HIDE_ALLOW_ORIGIN` override, and NO auth layer at all, so there is no
  injected-enforcement seam: an embedded host and a networked host cannot be
  given different auth or effect layers without editing the transport crate.
  (b) Authorization is by ORIGIN, not identity. CORS is the only thing between a
  local process and `/v1/hide/rpc`, and no non-browser client honors it; the
  donor's password-versus-embedded split at least models "the host chooses
  enforcement". (c) The surface is four generic endpoints, so there is nowhere to
  attach per-domain middleware the way the donor attaches location middleware per
  group; effect gating happens deeper, inside the host, which is safe but not
  inspectable at the boundary.
- Verdict: ALREADY-COVERED for provider neutrality, ADOPT-LATER for the
  injection seam (contract declares middleware slots, host supplies enforcement
  at bind time), REJECT nothing. Non-negotiable on any later adoption: an
  embedded host may drop a password but may NEVER skip the effect and sandbox
  layer. No code this pass, since changing the route surface would move a
  boundary the frontend pipeline is currently building against.

## Comparison 5: SDK generation (document only)

- What opencode does: `packages/httpapi-codegen` compiles one `HttpApi` to a
  neutral intermediate (`compile`), then emits two INDEPENDENT clients from it
  (`emitEffect`, `emitPromise`), each with its own type projection. It flattens
  path, query, header, and body fields into one input object and rejects
  duplicate field names, unwraps an exact `{ data: A }` success envelope, maps
  no-content success to void, rejects ambiguous multi-success contracts, and
  collapses transport, unexpected-status, and decode failures into one stable
  generated `ClientError`. Output is Prettier-formatted, tracked in
  `.httpapi-codegen.json` so regeneration removes only files the generator owns,
  committed for review, and CI regenerates and fails when the worktree changes.
- What HIDE does: `crates/hide-sdk` reads hide-protocol's schemars derivations
  and emits four artifacts from one binary (`protocol.schema.json`,
  `protocol.d.ts`, `command_catalog.json`, `commands.d.ts`) plus
  `fixtures/events.json`, all byte-compared against committed goldens in
  `tests/golden.rs`. Determinism is designed in (schemars definitions and
  properties are BTreeMap or BTreeSet backed, serde_json is built without
  `preserve_order`), which is what makes a byte comparison meaningful rather than
  flaky. There is one hand-written async client (`client.rs`) over a `Transport`
  trait with a `MockTransport` for tests.
- Gaps versus the donor: (a) no generated-file manifest, so a root type removed
  from `ROOT_TYPE_NAMES` leaves a stale golden that nothing deletes; the donor's
  `.httpapi-codegen.json` solves exactly this. (b) One emitter and one target:
  there is no compile-to-neutral-intermediate stage, so a second target (a Rust
  client, an OpenAPI document, another language) would re-walk the schema instead
  of sharing an intermediate. (c) The Rust client is hand-written, which applies
  the "no handwritten mirror types" rule to TypeScript but not to Rust. (d) No
  unified error projection: the SDK does not emit one stable client error shape
  the way the donor's `ClientError` does. (e) The golden test proves the
  committed artifact matches the code, but no CI job regenerates and fails on a
  dirty worktree, so a DELETED artifact is caught only by the compile of
  `include_str!`, not by a regeneration diff.
- Verdict: ALREADY-COVERED for the core discipline (one source, deterministic
  emit, committed artifacts, drift caught by a test), ADOPT-LATER for the
  generated-file manifest, REJECT the two-client emit and the Prettier
  dependency: HIDE has one consumer and deliberately no JS toolchain in the
  generation path. No code this pass; the goldens are frozen.

## Explicitly out of scope: LSP and language-server management

Mechanism 5 above (LSP integration) is NOT a product area this campaign opens.
HIDE may deepen its EXISTING diagnostics path later, but external language-server
management (discovering, installing, spawning, and supervising third-party server
binaries) is not introduced here, and its ledger row stays `deferred`. If it is
ever revisited, the spawn step must route through hide-security sandbox and the
Process effect gate: opencode spawns servers directly with no effect check, which
is precisely the assumption this campaign rejects.

## What landed in this pass

Two closures, both clean-room (no donor code copied, so no MIT notice attaches),
both integrating HIDE effect enforcement rather than donor permission semantics:

1. `crates/hide-compat/src/agents.rs`: `Agent::allows_tool`, the profile gate
   predicate (deny wins, empty allow list means inherit all), plus two tests.
2. `crates/hide-extension-registry/src/registry.rs`: `Registry::register` now
   refuses a manifest declaring Execute or Process with no sandbox isolation,
   plus a test (and one existing ranking-test fixture updated to declare the
   isolation its Execute effect implies).
