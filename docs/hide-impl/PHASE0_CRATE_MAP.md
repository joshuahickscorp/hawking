# Phase 0: Crate Map (target vs current, reconciled)

Edition: 2026-07-19 · Bible §71-72. Reconciles the Bible's 28-crate target repository shape against what the rehydrated packed backend already provides, so Phase 1+ reuses existing tested code instead of greenfielding, and only builds NEW crates where no packed crate owns the responsibility.

## 1. Current reality (after Phase 1a rehydration)

- **Retained runtime (6):** `hawking`, `hawking-core`, `hawking-serve`, `hawking-bench`, `hawking-seed-c`, `hawking-speculate`. These are the local model runtime and stay as the Model Substrate.
- **Rehydrated backend (13, from 5a99d0e2):** `hide-core`, `hide-serve`, `hide-backend`, `hide-kernel`, `hide-tools`, `hide-security`, `hide-fleet`, `hide-personalize`, `hawking-context`, `hawking-index`, `hawking-orch`, `hawking-research`, `hawking-eval`. Compiles clean; 392 tests green.

## 2. Target -> current mapping

Status: EXISTS (a packed crate owns it), EXTRACT (split out of a packed crate), CONSOLIDATE (merge packed pieces), NEW (build it), ARCHIVE.

| Bible target crate (§71) | Responsibility | Current owner | Status |
|---|---|---|---|
| `hide-protocol` | one schema source (types/events) | `hide-core` (api/event/ids) | EXTRACT (make the schema-gen authority, Bible §15.7) |
| `hide-agent-server` | thread/turn/item protocol, transports, backpressure | `hide-backend` (host) + `hide-serve` (transport) | CONSOLIDATE + EXTEND (Codex-inspired, Phase 2) |
| `hide-event-store` | append-only event log, replay | `hide-backend` (event bus + BackendReplayService) | EXTRACT |
| `hide-artifacts` | content-addressed artifact store | `hide-core` (BlobStore) + `hawking-research` (CAS) | CONSOLIDATE |
| `hide-workspace` | repos/environments/trust | `hide-core` + `hide-backend` | EXTRACT |
| `hide-environment` | exec context, fs roots, secrets, net policy | `hide-security` + `hide-backend` | EXTRACT/NEW |
| `hide-policy` | effect classes, allow/ask/deny engine | `hide-security` (permission engine) + `hide-core` (traits) | EXISTS -> EXTRACT |
| `hide-sandbox` | OS enforcement (Seatbelt/bubblewrap) | `hide-security` (Seatbelt render+spawn; OS enforcement is a seam) | EXISTS (logic) / NEW (enforcement) |
| `hide-kernel` | flat loop, plan-as-data, oracles | `hide-kernel` | EXISTS |
| `hide-agent-tree` | spawn/scheduler/roles | `hide-kernel` (subagents) + `hide-fleet` | EXTRACT/BUILD |
| `hide-scheduler` | resource-aware fleet scheduler | `hide-fleet` (FleetGovernor/queue) | EXISTS (REDESIGN) |
| `hide-context` | RepoMap, ContextPack, reserve-then-fill | `hawking-context` | EXISTS (rename) |
| `hide-index` | tree-sitter/merkle/FTS5/graph/retriever | `hawking-index` | EXISTS (rename) |
| `hide-memory` | outcome-governed memory | `hide-personalize` (records) + `hawking-context` (FTS5 store) | CONSOLIDATE |
| `hide-state` | state capsules (RWKV/KV), HTTP routes | RWKV state in `hawking-core` (unwired); no capsule crate | NEW (the moat; needs GPU readback + serve routes) |
| `hide-tools` | typed tools, edit applier, MCP client | `hide-tools` | EXISTS |
| `hide-program-runtime` | programmatic tool runtime (Book V) | none | NEW |
| `hide-edit` | transactional edit engine | `hide-tools` (verifying applier) | EXTRACT |
| `hide-shell` | shell + PTY | `hide-tools` (shell.run) | EXTRACT + NEW (PTY) |
| `hide-browser` | browser + computer use | none | NEW |
| `hide-verify` | oracles, static analysis, browser/visual | `hide-kernel` (ProcessOracle) | EXTRACT + EXTEND |
| `hide-eval` | pass@1 + Wilson CI | `hawking-eval` | EXISTS (rename) |
| `hide-extension-registry` | unified capability registry | none (packed pieces: tools/skills/hooks scattered) | NEW |
| `hide-mcp` | MCP host + client + server | `hide-tools` (JSON-RPC client) | EXISTS (client) -> EXTRACT + BUILD host/server |
| `hide-acp` | ACP server | none | NEW |
| `hide-sdk` | generated SDK | none | NEW (from `hide-protocol` schema) |
| `hide-cli` | TUI/CLI client | `hawking` (partial) | NEW (client over the agent server) |
| `hide-tauri` | desktop shell | `app/src-tauri` | EXISTS |
| (archive) | knowledge graph / arXiv | `hawking-research` | ARCHIVE (scope trap; drop from hide-backend deps in 1b) |

## 3. Reconciliation verdict

- **~15 of 28 target crates already have a working owner** in the rehydrated packed backend (kernel, context, index, tools, eval, policy/security, event-store, scheduler, artifacts, memory pieces). HIDE is a **reconnection + extension**, not a greenfield.
- **NEW crates** (no packed owner): `hide-program-runtime`, `hide-browser`, `hide-acp`, `hide-sdk`, `hide-extension-registry`, `hide-state` (capsules), and the elevated `hide-agent-server` protocol. These are the genuine build targets for Phases 2, 5, 7, 8, 9, 11.
- **Renames are deferred.** Phase 1 keeps the packed names (`hawking-context`, `hawking-index`, `hawking-eval`, `hawking-orch`) to avoid churn while wiring the vertical; the rename toward `hide-*` happens once the vertical is green and only where it reduces duplicate responsibility (Bible §71 "avoid duplicate responsibility").
- **hide-backend** is the current composition host (Bible calls it toward `hide-agent-server`); it stays as the host through Phase 1, then the protocol is elevated in Phase 2.

## 4. Dependency direction (Bible §72, enforced)

```
hide-core/protocol  <-  event/artifact store  <-  policy/sandbox  <-  workspace/environment
  <-  context/index/tools/verify/state  <-  kernel/agent-tree  <-  agent-server (hide-backend/hide-serve)  <-  clients (app/, hide-cli, acp)
```
Rules held: no UI crate owns backend truth (app/ is a client over `/v1/hide/*`); no model driver imports UI; no plugin gets ambient server authority (capability-scoped effects via `hide-security`). Verified: the rehydrated backend is HTTP-coupled to the runtime (no Rust dep on `hawking-core`/`hawking-serve`), which already satisfies "no model driver imports UI/backend".
