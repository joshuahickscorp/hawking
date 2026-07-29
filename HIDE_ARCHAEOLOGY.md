# HIDE live archaeology — subsystem classification

**Commit:** `fccb6b3036f94ad85453f130f394b8ff7be4bca9`  
**Method:** entry-points-inward static tracing + `cargo build` / `cargo test` on the HIDE crate set.  
**Scope:** read-only reconnaissance. Only this file and `HIDE_ARCHAEOLOGY.json` were written.

## Live entry points (enumerated first)

| Entry | Path | What it does |
|---|---|---|
| `hide-serve` bin | `crates/hide-serve/src/main.rs:20-40` | `BackendHost::open_workspace` → `hide_serve::router` → axum on `127.0.0.1:8744` |
| HTTP/WS routes | `crates/hide-serve/src/lib.rs:39-53` | `POST /v1/hide/intent` → `handle_intent`; `GET /v1/hide/events` (WS + catch-up); `POST /v1/hide/connector`; `POST /v1/hide/rpc`; `GET /healthz` |
| Tauri desktop | `app/src-tauri/src/main.rs` | Spawns `hide-serve` sidecar only (no direct host IPC) |
| `hide-acp-server` bin | `crates/hide-acp/src/bin/hide-acp-server.rs:19-30` | ACP stdio server with **`DeferredTurnHandler`** (no model, no backend host) |
| `hide-sdk-codegen` | `crates/hide-sdk/src/bin/hide-sdk-codegen.rs` | Codegen only (not a runtime host) |
| Hawking CLI / serve | `crates/hawking`, `hawking-serve` | Inference runtime; HIDE talks HTTP to it when supervised. (`hawking-seed-c` product-released under BC-BRIDGE-012 / B-RT5 — historical only) |

**Non-entry public APIs that tests call directly** (not a live FE/CLI/RPC path unless noted):

- `BackendHost::fleet_run` — unit-tested only
- `BackendHost::initialize` — integration-tested; **no hide-serve route**
- `BackendHost::evaluate_tool_policy` — unit-tested only
- `BackendHost::scrub_to_event` — used internally by checkpoint create/restore; not a custom intent
- `hide_tools::register_mcp_servers` — tested inside `hide-tools` only

---

## Per-subsystem classifications

### 1. backend host (`hide-backend/src/host.rs`, `rpc.rs`, `initialize.rs`)

- **verdict:** `REAL_WIRED` (with one transport hole on Initialize)
- **crates/files:** `crates/hide-backend/src/host.rs`, `rpc.rs`, `initialize.rs`, `commands.rs`, `services.rs`; consumed by `crates/hide-serve`
- **entry path:**
  1. `hide-serve` `main` → `BackendHost::open_workspace` (`host.rs:572`) → `from_services` (`host.rs:576`)
  2. `POST /v1/hide/intent` (`hide-serve/src/lib.rs:46`, handler ~199) → `BackendHost::handle_intent` (`host.rs:714`)
  3. `POST /v1/hide/rpc` (`lib.rs:51`) → `BackendHost::rpc` (`rpc.rs:154`)
  4. `POST /v1/hide/connector` → `BackendHost::call_connector` (`host.rs:2424`)
- **evidence:** Route table + host construction read; build of `hide-serve`/`hide-backend` succeeds; host unit tests exercise intent/rpc extensively.
- **missing link:** `BackendHost::initialize` (`host.rs:4532`) + `ConnectionRegistry` are implemented and unit/integration-tested, but **hide-serve has no Initialize route** and never calls `host.initialize` — capability negotiation is dead on the live transport.
- **tests:** Heavy unit coverage on host/rpc; integration tests (`wire_write_path`, `steer_and_dispatch`, etc.) hit the host through constructed `BackendHost`, not always through axum. Serve has its own route tests (`hide-serve` 10 passing).
- **confidence:** high

### 2. Planner / Executor / Verifier loop

- **verdict:** `PARTIAL`
- **crates/files:** `crates/hide-kernel` (`plan/`, `machine/driver.rs`, `verify/`); host glue `run_turn_kernel` / `build_turn_kernel` / `run_turn_core` in `host.rs`
- **entry path (partial):**
  - Default live path: `SubmitTurn` intent → `spawn_submit_turn_generation` (`host.rs:2316`) → **`run_turn_core` single-shot** (no Planner/tool loop) when `HIDE_KERNEL_TURN` is unset (`kernel_turn_enabled` at `host.rs:6124-6128`, default **off**)
  - Opt-in full loop: `HIDE_KERNEL_TURN=1` **and** supervised runtime `Ready` (`HIDE_MODEL_WEIGHTS` + `hawking serve`) → `build_turn_kernel` (`host.rs:2245`) → `run_turn_kernel` (`host.rs:6165`) → `AgentKernel::start_run` / `step` with `RuntimePlanner` + oracles + tool parse/dispatch
  - Offline / no model: publishes RuntimeStatus/Error; no PEV loop
- **evidence:** Comment at `host.rs:6116-6123` states kernel path is opt-in; `AgentKernel::new` still installs `StubPlanner` (fleet path); flagship kernel tests exist in `hide-kernel` and host.
- **missing link for full PEV on default serve:** (1) default `HIDE_KERNEL_TURN` off; (2) model supervisor only boots when `HIDE_MODEL_WEIGHTS` set (`host.rs:632-665`).
- **tests:** Kernel unit + `full_run` integration pass; host kernel-path tests pass in isolation. Unit tests do **not** prove default production env runs the loop.
- **confidence:** high

### 3. worktree fleets (`hide-fleet`)

- **verdict:** `REAL_UNWIRED`
- **crates/files:** `crates/hide-fleet` (`manager.rs`, `isolate.rs`, `queue.rs`, …); host method `BackendHost::fleet_run` (`host.rs:2707`)
- **entry path:** `NONE FOUND` from intent / rpc / connector / bin.
  - Closest live path: `create_worktree` custom intent (`host.rs:1185-1186`) → `spawn_worktree_add` — **plain `git worktree add`**, not `FleetManager` / hide-fleet isolation.
- **evidence:** `grep fleet_run` only hits `host.rs` definition + unit test `host_fleet_run_schedules_and_completes` and docs. No `HANDLED_CUSTOM_NAMES` entry for fleet. RPC has no fleet methods.
- **missing link:** No intent name, RPC method, or connector that calls `BackendHost::fleet_run` (or holds a long-lived `FleetManager` on the host).
- **tests:** Fleet crate tests pass (53); host unit test exercises `fleet_run` in-process only. Uses `AgentKernel::new` (StubPlanner) + `with_fake_worktrees()`.
- **confidence:** high

### 4. fleet governor

- **verdict:** `REAL_UNWIRED`
- **crates/files:** `crates/hide-fleet/src/scheduler.rs` (`FleetGovernor`); used by `FleetManager::new` (`manager.rs`)
- **entry path:** Only via `fleet_run` → `FleetManager::new(..., FleetGovernor::default(), ...)` (`host.rs:2723-2726`) → which itself has **no live entry** (see §3).
- **evidence:** No other host/serve call sites for `FleetGovernor`.
- **missing link:** Same as fleet: a live entry that constructs/runs `FleetManager` under real (non-test) traffic.
- **tests:** Scheduler/manager unit tests pass; not on wire path.
- **confidence:** high

### 5. merge funnel

- **verdict:** `REAL_UNWIRED`
- **crates/files:** `crates/hide-fleet/src/merge.rs` (`TournamentSelector`, `three_way_merge`, `integrate`, footprint planner)
- **entry path:** `NONE FOUND`. `FleetManager` does **not** call `merge::integrate` / `TournamentSelector`. Only `merge.rs` internal tests + a comment in `patterns.rs`.
- **evidence:** `grep integrate|TournamentSelector` outside `merge.rs` finds no production callers.
- **missing link:** Wire tournament/fan-out completion into `FleetManager` (or a host API) after parallel runs finish.
- **tests:** Module unit tests in `merge.rs` only.
- **confidence:** high

### 6. repository index (`hawking-index`)

- **verdict:** `PARTIAL`
- **crates/files:** `crates/hawking-index` (`query.rs` InMemory + Sqlite, `daemon.rs`, `semantic.rs`, …); host holds `BackendServices.code_index: Arc<InMemoryCodeIndex>` (`services.rs:1463`, always `InMemoryCodeIndex::default()` at `1518`)
- **entry path (working):**
  - Host construction always installs empty `InMemoryCodeIndex`
  - Connector `code_index` registered (`connectors.rs:782`) → `POST /v1/hide/connector` with methods `file.add_text` / `file.index` / `search`
  - Context compile on SubmitTurn uses `CodeIndexContextSource` over that index (`run_turn_kernel` / `run_turn_core`, e.g. `host.rs:6207`)
  - Kernel grounding: `Grounding::new(code_index)` in `build_turn_kernel` (`host.rs:2265`)
- **paths that do not work / are unused:**
  - `SqliteCodeIndex`, merkle daemon, hybrid retriever with live embeddings: **never selected by `BackendServices::open_workspace`**
  - Index starts **empty** unless something calls the connector to add files — no automatic workspace walk at open
- **evidence:** services construction; no `SqliteCodeIndex` reference under `hide-backend` outside comments/tests that still use InMemory.
- **missing link:** Durable index open + incremental daemon bind at `open_workspace`; optional auto-ingest of workspace.
- **tests:** hawking-index 44 unit tests pass (including Sqlite); host uses InMemory only.
- **confidence:** high

### 7. Context Stack / Context OS (`hawking-context`)

- **verdict:** `REAL_WIRED` (compiler + SQLite project memory on open; some KV/embed seams secondary)
- **crates/files:** `crates/hawking-context` (`compiler.rs`, `profiles.rs`, `sources.rs`, `memory.rs`, `kv.rs`, …)
- **entry path:**
  - `open_workspace` → `SqliteMemoryStore::open(.hide/memory/memory.db)` (`services.rs:1565-1569`) as Spine B Project Brain
  - Every accepted `SubmitTurn` (single-shot or kernel) → `ContextCompiler` + `CodeIndexContextSource` + optional repo instructions (`host.rs:6204-6220`, twin in `run_turn_core`)
  - Connector `context` registered for compile-from-index
  - Live ceiling snapshot via `HttpModelProvider::get_context_info` when model online
- **evidence:** Production functions in host (not test-only); context_manifest UiEvents published.
- **missing link (minor):** Live `KvStore` bridge to serve slots remains a seam; not required for the compile path to run.
- **tests:** 37 unit tests pass; host tests for compiled context on turn paths.
- **confidence:** high

### 8. session registry

- **verdict:** `REAL_WIRED`
- **crates/files:** `crates/hide-backend/src/services.rs` (`SessionRegistry`, durable via KV); used throughout host
- **entry path:**
  - `BackendServices::open` / `session()` → `SessionRegistry::open_or_create` (`services.rs:1590`)
  - Intents `new_session` / `open_session` (`host.rs:1189-1196`)
  - Fork / side-chat / checkpoint paths record ancestry via `sessions.record_session`
  - RPC `session/get`, `thread/get`, `thread/list` → `conversation_graph` (`rpc.rs:188-194`)
- **evidence:** Stable session across reopen unit test; live intents in `HANDLED_CUSTOM_NAMES`.
- **missing link:** none for core registry. RPC `session/new|list|close` still `NotImplemented` (`rpc.rs:381-386`) — lifecycle partly intent-only.
- **tests:** services + host + rpc tests cover registry.
- **confidence:** high

### 9. fork / time travel / rewind (`rewind.rs`, `replay.rs`)

- **verdict:** `REAL_WIRED` (fork + checkpoint rewind family); scrub as public intent is weak
- **crates/files:** `crates/hide-backend/src/replay.rs`, `rewind.rs`; host methods `fork_session*`, `scrub_to_event`, `checkpoint_rewind|replay|fork|inspect|restore`
- **entry path:**
  - `Intent::ForkSession` → `spawn_fork_session` (`host.rs:1143-1145`)
  - RPC `thread/fork` → `fork_session_from_event` (`rpc.rs:160-182`)
  - Custom intents `checkpoint_create|restore|rewind|replay|fork|compare|inspect` (`host.rs:817-831`, handler `handle_goal_checkpoint_intent` ~1550)
  - RPC `checkpoint/create|list|restore` (`rpc.rs:253-298`)
  - Internal: `checkpoint_create` uses `replay.scrub_to_event` (`host.rs:3934`)
- **evidence:** Multiple integration/unit tests for fork/rewind; methods on live intent + RPC surfaces.
- **missing link:** `scrub_to_event` is listed in command catalog (`commands.rs`) but **not** in `HANDLED_CUSTOM_NAMES` — clients cannot scrub via intent; only via internal checkpoint APIs / direct method.
- **tests:** replay/rewind modules + host checkpoint tests.
- **confidence:** high

### 10. tool parser and tool runner (`hide-tools` + kernel parse/runner)

- **verdict:** `PARTIAL`
- **crates/files:**
  - Implementations/registry: `crates/hide-tools` (`registry.rs`, `fs`, `edit`, `shell`, …)
  - Parser/runner: `crates/hide-kernel/src/tools/parse.rs`, `runner.rs`; used by `machine/driver.rs:371`
  - Host dispatcher: `crates/hide-backend/src/tools.rs` → `build_default_tool_dispatcher`
- **entry path (working without kernel):**
  - Host `from_services` builds registry + permission-gated dispatcher (`host.rs:578-586`)
  - Live effects: `save_file`, `RunCommand`, process intents, approved worktree, `dispatch_tool` → builtin tools
- **entry path (parser — conditional):**
  - Only inside kernel Act when `HIDE_KERNEL_TURN=1` and model emits tool text → `parse_tool_calls` → dispatcher (`driver.rs:370-374`)
- **evidence:** Default SubmitTurn does not parse tools; registry registers full catalog including git worktree tools.
- **missing link:** Default-on kernel turn (or another surface) so parse→run is not opt-in; MCP registration (see §12) for external tools.
- **tests:** hide-tools mostly pass (2 shell sandbox failures in this environment); kernel parse tests pass; host dispatcher/lease tests pass.
- **confidence:** high

### 11. memory (`hide-backend/src/memory.rs`)

- **verdict:** `REAL_WIRED`
- **crates/files:** `crates/hide-backend/src/memory.rs` (outcome-governed `MemoryLedger`); distinct from `hawking_context::MemoryStore` (Project Brain — also wired, §7)
- **entry path:**
  - Custom intents `memory_add|supersede|record_outcome|revalidate` (`host.rs:867-877`, `1223-1226`) → `handle_memory_workspace_env_intent` → `memory_add` etc. (`host.rs:4740+`)
  - Durable via host KV store
- **evidence:** `HANDLED_CUSTOM_NAMES` includes memory_*; host unit tests for revalidation/quarantine/supersede.
- **missing link:** none for ledger CRUD/revalidate. Automatic injection of Active records into every ContextCompiler pack is not fully proven as always-on (context path uses Project Brain + code index primarily).
- **tests:** host memory unit tests pass.
- **confidence:** high for ledger surface; medium for “always in compiled context”

### 12. MCP integration

- **verdict:** `REAL_UNWIRED`
- **crates/files:** `crates/hide-tools/src/mcp.rs` (`McpClient`, `register_mcp_servers`, tool mapping)
- **entry path:** `NONE FOUND` outside `hide-tools` unit tests.
- **evidence:** `register_mcp_servers` only referenced in `mcp.rs` itself. Host `build_default_tool_registry` / `register_builtin_tools` never attaches MCP servers. No intent/connector/config boot path.
- **missing link:** Call `register_mcp_servers` (or equivalent) from `BackendHost::from_services` / config load, and surface status to the client.
- **tests:** MCP client unit tests in hide-tools (stdio + HTTP fake) — unit only.
- **confidence:** high

### 13. ACP integration (`hide-acp`)

- **verdict:** `PARTIAL`
- **crates/files:** `crates/hide-acp` (protocol map, session, server, bin)
- **entry path:**
  - Live bin: `hide-acp-server` → `AcpServer::new(..., DeferredTurnHandler, ...)` (`hide-acp-server.rs:24-29`) → stdio JSON-RPC runs, but turns yield an honest **blocker** (no model, **no `BackendHost`**)
  - Mapping/handshake/projection: real and tested; crate does **not** depend on `hide-backend`
- **evidence:** Bin comment `DEFERRED_MODEL_REQUIRED`; `DeferredTurnHandler` in `server.rs:108+`.
- **missing link:** A `TurnHandler` that binds `BackendHost` / SubmitTurn (and model) into ACP `session/prompt`.
- **tests:** Integration tests `acp_boundary`, `acp_server`, `acp_wire` (24 total) pass with scripted/deferred handlers.
- **confidence:** high

### 14. serve path (`hide-serve`)

- **verdict:** `REAL_WIRED`
- **crates/files:** `crates/hide-serve/src/main.rs`, `lib.rs`
- **entry path:** bin → open workspace → router → axum; FE/Tauri target `127.0.0.1:8744`
- **evidence:** Built successfully; 10 serve tests pass; route table documents Wire-A/B + RPC + connector.
- **missing link:** Initialize handshake route (see §1); no fleet/MCP routes (correctly absent until wired in host).
- **tests:** serve crate tests pass.
- **confidence:** high

### 15. model/provider registry (`hide-backend/src/model_provider.rs`)

- **verdict:** `PARTIAL`
- **crates/files:** `crates/hide-backend/src/model_provider.rs` (`HttpModelProvider`); `supervisor.rs` (`RuntimeSupervisor`); orch `RoleRegistry` on services
- **entry path:**
  - If `HIDE_MODEL_WEIGHTS` set: `maybe_boot_runtime` (`host.rs:632`) spawns supervisor → when Ready, `runtime_base_url` feeds `HttpModelProvider` for SubmitTurn generation
  - Else: host stays headless; SubmitTurn surfaces model offline (no fake tokens)
  - Role/model metadata: `RoleRegistry::with_default_local_roles()` always present
- **evidence:** T5 HTTP-only design; unit tests against fake serve in supervisor testkit.
- **missing link:** Default production config that boots a real `hawking serve` without manual env; provider “registry” is thin (HTTP client + roles), not a multi-provider catalog UI/API.
- **tests:** supervisor + host generation tests; not end-to-end with real weights in this recon.
- **confidence:** high

### 16. permissions and effect adjudication (`hide-security`, `approval.rs`, `policy.rs`)

- **verdict:** `PARTIAL`
- **crates/files:** `crates/hide-security`; `hide-backend/src/security.rs`, `tools.rs` (dispatcher engine), `approval.rs`, `policy.rs`; gate book in host
- **entry path (wired):**
  - Every tool dispatch → `SecurityServices::permission_engine` via `build_*_tool_dispatcher` (`tools.rs:329+`)
  - Intent-level Ask policy + security gates (`handle_intent` hold/approve/deny_gate)
  - `approve_effect` / `deny_effect` → `ApprovalHub` (`host.rs:1135-1141`); drained in `run_turn_kernel` when paused
  - Process spawn uses sandbox helpers from hide-security
- **entry path (unwired / weak):**
  - `evaluate_tool_policy` / durable `policy.decision` ledger (`host.rs:3154`) — **only called from unit tests**, never from dispatcher or intent
  - ApprovalHub only affects turns that actually use the kernel path (`HIDE_KERNEL_TURN=1`)
- **evidence:** Dispatcher is load-bearing for save_file and agent tools; `grep evaluate_tool_policy` shows test-only callers.
- **missing link:** Invoke `evaluate_tool_policy` (or equivalent) on the real dispatch path if the ledger is product-required; default-on kernel for effect approvals to matter mid-turn.
- **tests:** policy unit tests; tools lease tests; hide-security 40 tests pass.
- **confidence:** high

### 17. verification (`hide-verify`)

- **verdict:** `PARTIAL`
- **crates/files:** `crates/hide-verify` (static analysis oracle, gate, receipts, review **profiles only**); host `run_static_analysis`, review receipt export, kernel `with_standard_oracles`
- **entry path (wired):**
  - Intent `run_static_analysis` → `handle_static_analysis_intent` → `hide_verify::StaticAnalysisOracle` (`host.rs:1301`, `1346-1380`)
  - Intent `export_review_receipt`
  - Kernel path (opt-in): standard deterministic oracles on tool dispatcher (`build_turn_kernel` → `with_standard_oracles`)
- **entry path (not live):**
  - Probabilistic Tier-4 reviewers: data only, `DEFERRED_MODEL_REQUIRED` (crate docs)
  - Kernel oracles not run on default single-shot SubmitTurn
- **evidence:** Crate docs + host intent arms + kernel builder.
- **missing link:** Default kernel turn so build/test oracles run post-edit; model-backed review roles if claimed.
- **tests:** hide-verify 15 integration tests pass; host static-analysis tests.
- **confidence:** high

### 18. durable sessions / checkpointing (`hide-state`)

- **verdict:** `REAL_UNWIRED` for the named crate **`hide-state`**; host has a **separate** REAL_WIRED checkpoint system
- **crates/files:**
  - Named target: `crates/hide-state` (capsule schema/store) — **not a dependency of hide-backend or hide-serve**
  - Actually durable checkpoints: `CheckpointStore` / `checkpoint_*` on `BackendHost` + KV in `services.rs` (event-log boundary checkpoints, not hide-state capsules)
- **entry path for hide-state:** `NONE FOUND`. Workspace member only. RPC `state/save|load|fork|release` returns `NotImplemented` (“durable state-capsule model is not built”) (`rpc.rs:457-464`).
- **entry path for host checkpoints (do not confuse):** intents + RPC checkpoint/* → `checkpoint_create/restore/...` — **REAL_WIRED**, implemented outside hide-state.
- **evidence:** `grep hide_state` only hits hide-state itself + a protocol comment pointing at future capsules.
- **missing link:** Depend on `hide-state`, implement capsule save/load against live runtime, map RPC `state/*` onto it. Or stop claiming hide-state as the live durable-session plane.
- **tests:** hide-state 26 tests pass in isolation (schema-only claim matches code).
- **confidence:** high

---

## Summary

### Counts per verdict

| Verdict | Count | Subsystems |
|---|---|---|
| `REAL_WIRED` | 6 | 1 backend host, 7 context, 8 session registry, 9 fork/rewind, 11 memory, 14 serve |
| `REAL_UNWIRED` | 5 | 3 worktree fleets, 4 fleet governor, 5 merge funnel, 12 MCP, 18 hide-state |
| `PARTIAL` | 7 | 2 PEV loop, 6 index, 10 tools parser/runner, 13 ACP, 15 model provider, 16 permissions/policy, 17 verification |
| `STUB` | 0 | — |
| `OBSOLETE` | 0 | — |
| `BLOCKED` | 0 | (ACP turn is deferred, not blocked by a missing crate) |
| `MISSING` | 0 | (claimed crates exist; wiring is the gap) |

### Shortest list of missing wires (highest leverage)

These are ordered to flip the most subsystems toward `REAL_WIRED` with the least new surface:

1. **Wire `FleetManager` to a live intent/RPC** (e.g. `fleet_run` / agent-spawn) and keep a host-owned manager  
   → flips **§3 worktree fleets** and **§4 fleet governor**.

2. **Call `merge::integrate` / tournament select from `FleetManager` on job completion**  
   → flips **§5 merge funnel** (depends on 1).

3. **Call `hide_tools::register_mcp_servers` from host boot/config**  
   → flips **§12 MCP**.

4. **Default `HIDE_KERNEL_TURN=1` once ready (or make kernel the non-env default when model Ready)**  
   → upgrades **§2 PEV**, **§10 tool parser/runner**, **§16 approval mid-turn**, **§17 kernel oracles** from partial toward wired-by-default.

5. **Bind `SqliteCodeIndex` (+ optional daemon) in `BackendServices::open_workspace`** and auto-ingest workspace  
   → upgrades **§6 index** from empty InMemory partial to durable REAL_WIRED.

6. **hide-serve route for Initialize → `BackendHost::initialize`**  
   → closes the hole in **§1**.

7. **ACP `TurnHandler` → BackendHost SubmitTurn**  
   → upgrades **§13 ACP**.

8. **Either implement RPC `state/*` on `hide-state` capsules or demote the crate in doctrine**  
   → resolves **§18** (today host checkpoints are already wired under a different schema).

9. **Optional: call `evaluate_tool_policy` on dispatch** if the policy ledger is product truth  
   → completes **§16** policy half.

### Docs that the code contradicts

| Document claim | Code reality |
|---|---|
| `docs/hide-bible/SCAFFOLD_STATUS.md`: hide-backend “fleet wired”, solid 5/5; hide-fleet “schedule_tick launching real kernel runs” as completed scaffold | `fleet_run` exists but **no live entry** reaches it; FE uses plain `create_worktree` git, not FleetManager |
| Same doc: hide-tools “real MCP (stdio + HTTP)” as part of the completed backend | MCP client is real **in-crate**; host never registers MCP servers → **unwired** |
| Same doc: “agent loop … complete”; “no further development on the agent loop” | Full Planner→Executor→Verifier is **opt-in** (`HIDE_KERNEL_TURN`) and model-gated; default SubmitTurn is single-shot |
| Same doc: hawking-index durable Sqlite + daemon as the standing organ | Host only constructs **empty `InMemoryCodeIndex`**; Sqlite/daemon unused by host |
| Scaffold narrative that hide-state is part of durable agent state plane | `hide-state` is schema-only, not linked; RPC state/* `NotImplemented`; real checkpoints live in host KV/event-log |
| Consolidation tone that “load-bearing hide-fleet dep” equals product reachability (`hide-backend` lib.rs docs) | Dependency + method are real; **reachability from transport is not** |

Treat bible/scaffold docs as **claims**. Prior “vertical slice works” language overstates default-path wiring.

### Workspace build and tests

**Build (key HIDE set):**  
`cargo build -p hide-serve -p hide-backend -p hide-fleet -p hide-tools -p hide-verify -p hide-state -p hide-acp -p hawking-context -p hawking-index -p hide-kernel`  
→ **succeeds** (`Finished dev` ~27s) on commit `fccb6b30`. One non-fatal `strand-quant` `cfg(kani)` warning.

**Tests (HIDE-focused packages, this machine):**

| Package | Result |
|---|---|
| hawking-context | 37 passed |
| hawking-index | 44 passed |
| hide-acp (lib + 3 integration) | 24 passed |
| hide-backend lib | **200 passed, 2 failed** |
| hide-backend integration (7 test bins) | 33 passed |
| hide-fleet | 53 passed |
| hide-tools lib | **60 passed, 2 failed** |
| hide-verify | 15 passed |
| hide-state | 26 passed |
| hide-kernel | 74 passed |
| hide-security | 40 passed |
| hide-serve | 10 passed |

**Totals (this recon):** **616 passed, 4 failed** across the packages above.

**Failing tests (environment / process, not used as wiring proof):**

1. `hide-backend::host::tests::host_records_run_command_intent_and_executes_command_api` — expected stdout `"api"`, got `""`
2. `hide-backend::host::tests::trace_d_service_process_persists_streams_and_captures` — service process heartbeats not observed
3. `hide-tools::shell::tests::shell_run_executes_and_captures_stdout`
4. `hide-tools::shell::tests::shell_run_nonzero_exit_is_ok_data`

These look like sandbox/process execution environment issues in the recon host, not evidence that the dispatch tables are missing. They **do** mean “all tests green” is **false** for a clean gate.

Full `cargo test --workspace` (hawking-core Metal, etc.) was **not** required to classify HIDE wiring and was not completed as a single green run.

---

## How to read this map for construction

- Prefer **wiring** the five `REAL_UNWIRED` subsystems over reimplementing them.
- Prefer **enabling** the seven `PARTIAL` default paths over new scaffolds.
- Do **not** treat “crate compiles + unit tests pass” as done — that was the failure mode this pass was designed to catch.
