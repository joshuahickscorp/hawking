# HIDE Archaeology v2

**Schema:** `hawking.hide.archaeology.v2`  
**Commit:** `d9bca273e86ca189a7b550f80734db7bf9ca94ad`  
**At:** 2026-07-27T02:10:00Z  
**Method:** entry-points-inward (live bins/routes → subsystems). Old `HIDE_ARCHAEOLOGY.json` at `fccb6b30` is **stale and was not trusted**.

**Worktree write status:** Repository root is **not writable** from this sandbox (`com.apple.macl` / Operation not permitted). Deliverables written to:

- `/tmp/hide-archaeology-v2/HIDE_ARCHAEOLOGY_V2.{md,json}`
- `/private/tmp/claude-503/-Users-scammermike-Downloads-hawking/34fc1e25-1b17-421d-8f66-207106fd8ba3/scratchpad/HIDE_ARCHAEOLOGY_V2.{md,json}`

Copy into the repo root when the MACL restriction is lifted.

---

## Why v2

Five wiring increments after `fccb6b30`:

| Commit | Title |
|--------|--------|
| `8682e8a6` | MCP registration, fleet dispatch, merge funnel, SQLite index |
| `5309baa8` | Effectful-step approval round-trip |
| `35be6027` | Permission hole, initialize route, hide-state absorbed |
| `8ec49108` | Context OS capability honesty, rot detection, reversible compaction |
| `def433ff` / `30958bcc` | Live model turn loop → real tokens end to end |

---

## Live entry points

| Entry | Path | Role |
|-------|------|------|
| **hide-serve** | `crates/hide-serve/src/main.rs:20-40` | `BackendHost::open_workspace` + axum `127.0.0.1:8744` |
| **routes** | `crates/hide-serve/src/lib.rs:43-64` | intent, events, connector, rpc, **initialize** (new) |
| **Tauri** | `app/src-tauri/src/main.rs:19-72` | hide-serve sidecar only |
| **hide-acp-server** | `crates/hide-acp/src/bin/hide-acp-server.rs:23-39` | `BackendHost` + `BackendTurnHandler` (not deferred) |
| **hawking serve** | `crates/hawking`, `hawking-serve` | inference; HIDE reaches via HTTP when Ready |

Routes at HEAD:

- `GET /healthz`
- `POST /v1/hide/intent` → `handle_intent`
- `GET /v1/hide/events` (WS + catch-up)
- `POST /v1/hide/connector`
- `POST /v1/hide/rpc`
- `POST /v1/hide/initialize` → `host.initialize`

---

## Summary counts

| Verdict | Count |
|---------|------:|
| REAL_WIRED | 19 |
| PARTIAL | 17 |
| REAL_UNWIRED | 4 |
| MISSING | 1 |
| OBSOLETE | 1 |
| STUB / BLOCKED | 0 |

**Headline:** v1’s top REAL_UNWIRED set is largely closed (MCP, fleet entry, merge call, SQLite index, ACP host, initialize, approval). Remaining mass is PARTIAL: fixture-bound fleet isolation, StubEmbedding semantic leg, heuristic token budget, opt-in kernel, and doctrine-vs-store gaps (six memory classes).

---

## Subsystems

### REAL_WIRED

| Subsystem | Entry path (abbrev) | Evidence |
|-----------|---------------------|----------|
| **Backend host** | serve main → `open_workspace` → router → handle_intent/rpc/initialize | Initialize + ConnectionRegistry live |
| **Context Stack / Context OS** | `run_turn_core` / `run_turn_kernel` → ContextCompiler + capability + rot | `8ec49108` on live path |
| **context-rot detection** | `detect_context_rot` post-compile / post-turn → UiEvent | host.rs 6242/6800/7239 |
| **reversible compaction** | compiler archive `compacted_from` on live compile | unit + production compiler |
| **repository index + symbol graph** | `bind_workspace_code_index` Sqlite + ingest | `8682e8a6` W4 |
| **lexical retrieval** | FTS5 via `CodeIndex::search` on every compile | load-bearing |
| **SQLite index** | `.hide/index/code.sqlite` at open | same as index |
| **session registry** | SessionRegistry open-or-create | intents + rpc + ACP |
| **Chat/IDE shared session identity** | one host registry; FE ipc + ACP SubmitTurn | store adopts `session_id` from events |
| **fork / time travel** | ForkSession, side_chat, checkpoint_*, scrub | event-log prefix copy |
| **semantic checkpoints** | checkpoint_* + RPC state/* → CheckpointStore | hide-state superseded |
| **permission + approval round-trip** | PermissionEngine + approve_effect/deny_effect drain | `5309baa8` + step_id hole closed |
| **MCP** | `register_mcp_servers_at_boot` from `from_services` | `.hide/mcp.json` |
| **ACP** | BackendTurnHandler → `handle_intent` SubmitTurn | no DeferredTurnHandler |
| **durable goals** | goal_* → GoalStore | durable KV |
| **canonical event stream** | JsonlEventLog + UiEventBus + /events | single authority |
| **Git/worktree tools** | builtin git.* + create_worktree intent | real git |
| **shell/process continuity** | shell.run + ProcessSupervisor artifacts | seatbelt (tests env-fragile) |
| **artifact control** | capture_process_artifact + blob store | live intents |

### PARTIAL

| Subsystem | Flag / fixture | Missing link |
|-----------|----------------|--------------|
| **Live model turn** | `HIDE_MODEL_WEIGHTS` default unset | Operational weights; code path is real HttpModelProvider → hawking `/v1/hawking/generate` SSE (not canned) |
| **PEV kernel loop** | `HIDE_KERNEL_TURN` default off | Prereqs **met** (approval + answer); **do not** flip default as sole fix — soak first |
| **tokenizer-true budgeting** | heuristic `chars/4` | `with_counter(TokenCounter::from_file(...))` at boot |
| **semantic retrieval** | `StubEmbeddingClient` always | `HttpEmbeddingClient` when Ready |
| **six memory classes** | 5 kinds, **one** Sqlite store | User/Verification are not MemoryKind stores — demote doctrine or add stores |
| **tool parser / runner** | kernel-gated | same as kernel promotion (not default flip alone) |
| **tool/effect registry** | 9-field doctrine incomplete | parallel_safe + rollback not first-class; receipt is events not ToolSpec |
| **fleet dispatch / governor / worktree fleets** | `with_fake_worktrees`, `FixedResourceProbe`, `AgentKernel::new` | real worktrees + OsResourceProbe + real launcher |
| **merge funnel** | synthetic footprints | real diffs + oracles into CandidatePatch |
| **independent verification** | static analysis yes; VerifierBranch taxonomy only | live mint of verifier branch |
| **restart recovery** | durable recover yes; auto re-drive no | resume interrupted runs |
| **subagents** | side_chat yes; agent/* NotImplemented | implement or demote |
| **tests/build/debugger tools** | proc tools yes; debugger no; oracles kernel-gated | — |
| **skills/hooks** | compat parse → context; no executor | register skill/hook runtime |

### REAL_UNWIRED

| Subsystem | Note |
|-----------|------|
| **change-aware retrieval** | Index daemon exists; host never starts it |
| **prefix/KV reuse** | HttpKvStore exists; no host caller |
| **copy-on-write warm forks** | KV APIs only; session fork is different and REAL_WIRED |
| **browser/retrieval tools** | hide-browser ReplayDriver only; not a host dep |

### MISSING / OBSOLETE

| Subsystem | Verdict |
|-----------|---------|
| **notebooks** | MISSING |
| **hide-state capsules** | OBSOLETE (CheckpointStore is authority) |

---

## Live model tokens — proof

Path:

1. `Intent::SubmitTurn` → `spawn_submit_turn_generation` (`host.rs:2452`)
2. `maybe_boot_runtime` if `HIDE_MODEL_WEIGHTS` (`host.rs:725`)
3. Default: `generate_submit_turn` → `run_turn_core` (`host.rs:6873`)
4. `HttpModelProvider::generate` → `POST {base}/v1/hawking/generate` SSE (`model_provider.rs`)
5. Token batches on Wire-B; assistant text persisted to event log

**Not canned:** provider is supervised `hawking serve`. Integration `tests/live_model_turn.rs` asserts non-empty assistant text. This recon observed a live `hawking serve` spawn with `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` when weights were discoverable. With `HIDE_LIVE_MODEL_TURN=0` the test skips cleanly.

Verdict remains **PARTIAL** because generation is env-gated (defaults offline), not because the path fabricates tokens.

---

## HIDE_KERNEL_TURN prerequisites

Comment at `host.rs:6294-6296` still says the kernel stays opt-in until **approval round-trip** and **answer surfacing** land.

| Prerequisite | Met? | Evidence |
|--------------|------|----------|
| Approval round-trip (effectful-step resume) | **Yes** | `approve_effect`/`deny_effect` → ApprovalHub; drained in `run_turn_kernel`; step_id required; tests ~host.rs:10196 |
| Answer surfacing | **Yes** | Kernel turn publishes non-empty assistant answer; tests ~host.rs:8940/9055 |

**Do not recommend flipping the default.** Promotion needs a separate soak under live model + tools, not a default flip.

---

## Six memory classes — stores vs labels

| Claimed class | Reality |
|---------------|---------|
| working | `MemoryKind::Working` **label** on one Sqlite table |
| episodic | `MemoryKind::Episodic` label |
| semantic-project | `Semantic` + `Project` labels (not separate DBs) |
| procedural | `MemoryKind::Procedural` label |
| user | `MemoryScope::User` on **MemoryLedger** (KV claims), not MemoryKind store |
| verification | **hide-verify receipts**, not a memory store |

**One** durable Project Brain (`SqliteMemoryStore` at `.hide/memory/memory.db`) + **one** claim ledger on KV. Labels do not count as six stores → **PARTIAL**.

---

## Tool/effect registry — nine fields

| Field | Exists? | Enforced? |
|-------|---------|-----------|
| schema | yes (`input_schema`) | progressive load in extension-registry |
| effects | yes (`EffectSet` / registry `Effect`) | permission uses declared effects |
| permission | yes | `ToolDispatcher` + PermissionEngine |
| sandbox | yes (shell seatbelt) | fail-closed off-macOS unless opted out |
| parallel safety | **no dedicated field** | only open_world / process policy |
| idempotence | yes (`annotations.idempotent`, `idempotency_key`) | annotated |
| timeout | yes (`timeout_ms`) | shell/proc watchdog |
| rollback | **no tool field** | host diff/rewind only |
| receipt | **partial** | `tool.call`/`tool.result` events; not ToolSpec field |

---

## Supersession table (vs v1 at fccb6b30)

| Subsystem | Old | New | Commit | Evidence |
|-----------|-----|-----|--------|----------|
| MCP | REAL_UNWIRED | **REAL_WIRED** | 8682e8a6 | `register_mcp_servers_at_boot` |
| worktree fleets | REAL_UNWIRED | **PARTIAL** | 8682e8a6 | `fleet_run` intent; still fakes |
| fleet governor | REAL_UNWIRED | **PARTIAL** | 8682e8a6 | on path; FixedResourceProbe |
| merge funnel | REAL_UNWIRED | **PARTIAL** | 8682e8a6 | `merge_terminal_jobs`; synthetic footprints |
| repository index | PARTIAL | **REAL_WIRED** | 8682e8a6 | SqliteCodeIndex at open |
| ACP | PARTIAL | **REAL_WIRED** | 35be6027 | BackendTurnHandler |
| hide-state | REAL_UNWIRED | **OBSOLETE** | 35be6027 | superseded by CheckpointStore |
| initialize hole | missing_link | **closed** | 35be6027 | POST `/v1/hide/initialize` |
| permissions | PARTIAL | **REAL_WIRED** | 5309baa8+35be6027 | approval round-trip + step_id |
| live model turn | PARTIAL | PARTIAL | def433ff | real tokens proven; still env-gated |
| PEV kernel | PARTIAL | PARTIAL | 5309baa8 | prereqs met; default still off |
| memory (six-class claim) | REAL_WIRED (collapsed) | **PARTIAL** | reclassify | not six stores |

---

## Facades

1. **`docs/hide-bible/SCAFFOLD_STATUS.md:4`** — “agent loop complete / no further development” while PEV is opt-in.
2. **Same doc :33** — hide-fleet “real kernel runs / merge solid 5/5” while fake worktrees + synthetic merge.
3. **Same doc :28** — index “Sqlite + daemon standing organ”; daemon still unstarted.
4. **`hide-backend/src/lib.rs:18-19`** — fleet “load-bearing” overstates isolation quality.
5. **`services.rs:1821-1831`** — `capabilities.fleet=true` with fixture-bound fleet_run.
6. **`hawking-index/query.rs:614-633`** — hybrid search always uses StubEmbeddingClient.
7. **`host.rs:6294-6296`** — stale comment that approval/answer have not landed (they have).
8. **Six-store memory narrative** vs one Sqlite + kind column.
9. **hide-state** as live state plane — crate itself now says superseded.

---

## Build / test truth

### Build

```
CARGO_TARGET_DIR=/tmp/hide-archaeology-v2-target cargo build --workspace
→ exit 0 (~27s; strand-quant cfg(kani) warning)
```

Default `target/debug/.cargo-lock` was Operation not permitted in sandbox; alternate target dir used.

### Tests (verbatim-style summaries)

| Package | Passed | Failed | Exit |
|---------|-------:|-------:|-----:|
| hide-core | 12 | 0 | 0 |
| hide-serve | 11 | 0 | 0 |
| hide-backend lib | 212 | 2 | 101 |
| hide-backend integration | ~33 | 1 | mixed |
| hide-kernel | 73 | 1 | 101 |
| hide-tools | 60 | 2 | 101 |
| hide-security | 40 | 0 | 0 |
| hide-fleet | 55 | 0 | 0 |
| hide-personalize | 38 | 0 | 0 |
| hawking-context | 45 | 0 | 0 |
| hawking-index | 44 | 0 | 0 |
| hide-compat | 39 | 0 | 0 |
| hide-extension-registry | 33 | 0 | 0 |
| hide-state | 26 | 0 | 0 |
| hide-protocol | 27 | 0 | 0 |
| hide-program-runtime | 34 | 0 | 0 |
| hide-browser | 19 | 0 | 0 |
| hide-acp | 25 | 0 | 0 |
| hide-sdk | 18 | 0 | 0 |
| hide-verify | 15 | 0 | 0 |
| **approx total** | **~859** | **6** | |

**Failures (not fixed — read-only audit):**

1. `hide-backend::host_records_run_command_intent_and_executes_command_api` — empty stdout
2. `hide-backend::trace_d_service_process_persists_streams_and_captures` — no heartbeats
3. `hide-tools::shell_run_executes_and_captures_stdout`
4. `hide-tools::shell_run_nonzero_exit_is_ok_data` (exit 71 vs 3 — sandbox)
5. `hide-kernel::full_run::failing_real_oracle_triggers_repair`
6. `first_model_free_implementation_receipt` — `PermissionDenied` in sandbox

Shell/process failures look like seatbelt/env issues (same class as v1 recon). Full green gate is false.

---

## Ranked missing wires (leverage / effort)

1. **Fleet production isolation** — drop `with_fake_worktrees`, use `OsResourceProbe`, real kernel launcher → fleet dispatch / governor / worktrees.
2. **Merge real footprints** — feed diffs/oracles into `merge_terminal_jobs` (depends on 1).
3. **Semantic embeddings** — `HttpEmbeddingClient` when Ready.
4. **Tokenizer-true budget** — install real `TokenCounter` at boot.
5. **Kernel promotion criteria** — prereqs met; **do not flip `HIDE_KERNEL_TURN` default**; define soak, then promote.
6. **Prefix/KV reuse** — `HttpKvStore` on turn start (+ CoW warm forks).
7. **Change-aware index** — start living-index daemon at open.
8. **Six memory classes** — implement or demote doctrine.
9. **Subagents** — implement `agent/*` or demote to side_chat.
10. **Browser / notebooks / skills** — CDP + host; notebook decision; skill/hook executor.

---

## Companion JSON

See `HIDE_ARCHAEOLOGY_V2.json` (same directories as this file) for full per-subsystem records: `name`, `verdict`, `crates`, `entry_path`, `evidence`, `missing_link`, `fixtures_in_place`, `tests`, `confidence`.
